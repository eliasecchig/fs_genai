"""Vector Store in Google Cloud BigQuery."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from functools import partial
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple, Literal
import time
import numpy as np
import pandas as pd
from google.api_core.exceptions import ClientError, ServiceUnavailable
from langchain_community.vectorstores.utils import DistanceStrategy, maximal_marginal_relevance
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.runnables.utils import Input, Output
from langchain_core.runnables.config import RunnableConfig
from langchain_google_community._utils import get_client_info
from pydantic import BaseModel, validate_call
from vertexai.resources.preview import FeatureOnlineStore, FeatureView, FeatureViewBigQuerySource
from vertexai.resources.preview.feature_store import utils
from google.cloud.aiplatform_v1beta1.types import feature_online_store_service
from google.cloud import bigquery

from google.cloud.exceptions import NotFound
from typing import Union
from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreAdminServiceClient, FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1beta1.types import NearestNeighborQuery
from google.cloud.aiplatform_v1beta1.types import feature_online_store as feature_online_store_pb2
from google.cloud.aiplatform_v1beta1.types import \
    feature_online_store_admin_service as feature_online_store_admin_service_pb2
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple
)

_vector_table_lock = Lock()  # process-wide BigQueryVectorSearch table lock


class BaseExecutor(BaseModel):
    def sync(self):
        raise NotImplementedError()

    def similarity_search_by_vectors_with_scores_and_embeddings(self,
                                                                embeddings: List[float],
                                                                k: int = 5):
        raise NotImplementedError()

    def set_vector_store(self, vector_store: VertexFeatureStore):
        self._vector_store = vector_store


class BruteForceExecutor(BaseExecutor):
    type: Literal["brute_force"] = "brute_force"

    def set_vector_store(self, vector_store: VertexFeatureStore):
        self._vector_store = vector_store
        self.sync()

    def sync(self):
        self._df = self._query_table_to_df()
        self._vectors = np.array(self._df[self._vector_store.text_embedding_field].tolist())
        self._vectors_transpose = self._vectors.T
        self._df_records = self._df.drop(columns=[self._vector_store.text_embedding_field]).to_dict("records")

    def similarity_search_by_vectors_with_scores_and_embeddings(
            self,
            embeddings: List[float],
            k: int = 5,
    ) -> List[List[Document, float, List[float]]]:
        num_queries = len(embeddings)
        scores = embeddings @ self._vectors_transpose
        sorted_indices = np.argsort(-scores)[:, :k]
        results = [np.array(self._df_records)[x] for x in sorted_indices]
        top_scores = scores[np.arange(num_queries)[:, np.newaxis], sorted_indices]
        top_embeddings = self._vectors[sorted_indices]
        # TODO add metadata filtering
        documents = []
        for query_results, query_scores, embeddings_results in zip(results, top_scores, top_embeddings):
            query_docs = []
            for doc, doc_score, embedding in zip(query_results, query_scores, embeddings_results):
                query_docs.append(
                    [Document(page_content=doc[self._vector_store.content_field], metadata=doc), doc_score, embedding])
            documents.append(query_docs)
        return documents

    def _query_table_to_df(self):
        from google.cloud import bigquery
        client = bigquery.Client(project=self._vector_store.project_id)
        metadata_fields = list(self._vector_store.extra_fields.keys())
        metadata_fields_str = ", ".join(metadata_fields)

        table = f"{self._vector_store.project_id}.{self._vector_store.dataset_name}.{self._vector_store.table_name}"
        fields = f"{self._vector_store.doc_id_field}, {self._vector_store.content_field}, {self._vector_store.text_embedding_field}, {metadata_fields_str}"
        query = f"""
        SELECT {fields}
        FROM {table}
        """
        # Create a query job to read the data
        print(f"Reading data from {table}. It might take a few minutes...")
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE
        )
        query_job = client.query(query, job_config=job_config)
        df = query_job.to_dataframe()
        return df


class BigQueryExecutor(BaseExecutor):
    type: Literal["big_query"] = "big_query"
    distance_strategy: str = DistanceStrategy.EUCLIDEAN_DISTANCE


    def model_post_init(self, __context: Any) -> None:
        self._creating_index = False
        self._have_index = False
        self._last_index_check = datetime.min

    def sync(self):
        self._initialize_bq_vector_index()

    def _initialize_bq_vector_index(self, min_index_row=5, index_check_secs=60) -> Any:
        """
        A vector index in BigQuery table enables efficient
        approximate vector search.
        """

        if self._have_index or self._creating_index:
            # Already have an index or in the process of creating one.
            return
        table = self._vector_store._bq_client.get_table(self._vector_store.full_table_id)
        if (table.num_rows or 0) < min_index_row:
            # Not enough rows to create index.
            self._logger.debug("Not enough rows to create a vector index.")
            return
        if (
                datetime.utcnow() - self._last_index_check
        ).total_seconds() < index_check_secs:
            return
        with _vector_table_lock:
            if self._creating_index or self._have_index:
                return
            self._last_index_check = datetime.utcnow()
            # Check if index exists, create if necessary
            check_query = (
                f"SELECT 1 FROM `{self._vector_store.project_id}.{self._vector_store.dataset_name}"
                ".INFORMATION_SCHEMA.VECTOR_INDEXES` WHERE"
                f" table_name = '{self._vector_store.table_name}'"
            )
            job = self._vector_store._bq_client.query(
                check_query, api_method=bigquery.enums.QueryApiMethod.QUERY
            )
            if job.result().total_rows == 0:
                # Need to create an index. Make it in a separate thread.
                self._create_bq_index_in_background()
            else:
                self._vector_store._logger.debug("Vector index already exists.")
                self._have_index = True

    def _create_bq_index_in_background(self):  # type: ignore[no-untyped-def]
        if self._have_index or self._creating_index:
            # Already have an index or in the process of creating one.
            return
        self._creating_index = True
        self._vector_store._logger.debug("Trying to create a vector index.")
        thread = Thread(target=self._create_bq_index, daemon=True)
        thread.start()

    def _create_bq_index(self, min_index_row=5):  # type: ignore[no-untyped-def]
        table = self._vector_store._bq_client.get_table(self._vector_store.full_table_id)
        if (table.num_rows or 0) < min_index_row:
            # Not enough rows to create index.
            return
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            distance_type = "EUCLIDEAN"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            distance_type = "COSINE"
        # Default to EUCLIDEAN_DISTANCE
        else:
            distance_type = "EUCLIDEAN"
        index_name = f"{self._vector_store.table_name}_langchain_index"
        try:
            sql = f"""
                CREATE VECTOR INDEX IF NOT EXISTS
                `{index_name}`
                ON `{self._vector_store.full_table_id}`({self._vector_store.text_embedding_field})
                OPTIONS(distance_type="{distance_type}", index_type="IVF")
            """
            self._vector_store._bq_client.query(sql).result()
            self._have_index = True
        except ClientError as ex:
            self._vector_store._logger.debug("Vector index creation failed (%s).", ex.args[0])
        finally:
            self._creating_index = False

    def similarity_search_by_vectors_with_scores_and_embeddings(
            self,
            embeddings: List[List[float]],
            k: int = 5,
            batch_size: int = 100
    ) -> List[Tuple[Document, List[float], float]]:

        final_results = []
        for start in range(0, len(embeddings), batch_size):
            end = start + batch_size
            embs_batch = embeddings[start: end]
            final_results.extend(self._search_embeddings(embeddings=embs_batch, k=k))
        documents = []
        fields = [x for x in final_results[0].keys() if
                  x not in [self._vector_store.text_embedding_field, self._vector_store.content_field]]
        for result in final_results:
            metadata = {}
            for field in fields:
                metadata[field] = result[field]
            documents.append(
                [Document(page_content=result[self._vector_store.content_field], metadata=metadata), metadata["score"],
                 result[self._vector_store.text_embedding_field]])
        results_chunks = [documents[i * k:(i + 1) * k] for i in range(len(embeddings))]
        return results_chunks

    def _search_embeddings(self, embeddings, k=5):
        embeddings_query = "with embeddings as (\n"
        for i in range(len(embeddings)):
            if i != 0:
                embeddings_query += "\nUNION ALL\n"
            embeddings_query += f"SELECT {i} as row_num, @emb_{i} AS text_embedding"
        embeddings_query += "\n)\n"

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter(f"emb_{i}", "FLOAT64", emb)
                for i, emb in enumerate(embeddings)
            ],
            use_query_cache=True,
            priority=bigquery.QueryPriority.INTERACTIVE
        )
        full_query = embeddings_query + f"""
        SELECT
            base.*,
            query.row_num,
            distance AS score
        FROM VECTOR_SEARCH(
            TABLE `{self._vector_store.project_id}.{self._vector_store.dataset_name}.{self._vector_store.table_name}`,
            "text_embedding",
            (SELECT row_num, {self._vector_store.text_embedding_field} from embeddings),
            distance_type => "EUCLIDEAN",
            top_k => {k}
        )
        ORDER BY row_num, score
        """
        results = self._vector_store._bq_client.query(full_query, job_config=job_config,
                                                      api_method=bigquery.enums.QueryApiMethod.QUERY)
        return list(results)


class FeatureOnlineStoreExecutor(BaseExecutor):
    type: Literal["feature_online_store"] = "feature_online_store"
    online_store_name: str = None
    view_name: str = None
    online_store_type: str = "bigtable"
    cron_schedule: str = None
    location: str = None

    def set_vector_store(self, vector_store: VertexFeatureStore):
        self._vector_store = vector_store
        self.init_feature_store()

    def init_feature_store(self):
        if self.online_store_name is None:
            self.online_store_name = self._vector_store.dataset_name
        if self.view_name is None:
            self.view_name = self._vector_store.table_name
        if self.location is None:
            self.location = self._vector_store.location
        self._admin_client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": f"{self._vector_store.location}-aiplatform.googleapis.com"}
        )
        self._online_store = self._create_online_store()
        self._search_client = self._get_search_client()
        self._feature_view = self._create_feature_view()

    def sync(self):
        sync_response = self._admin_client.sync_feature_view(
            feature_view=f"projects/{self._vector_store.project_id}/locations/{self._vector_store.location}/featureOnlineStores/{self.online_store_name}/featureViews/{self.view_name}"
        )
        while True:
            feature_view_sync = self._admin_client.get_feature_view_sync(
                name=sync_response.feature_view_sync
            )
            if feature_view_sync.run_time.end_time.seconds > 0:
                status = "Succeed" if feature_view_sync.final_status.code == 0 else "Failed"
                print(f"Sync {status} for {feature_view_sync.name}.")
                # wait a little more for the job to properly shutdown
                time.sleep(30)
                break
            else:
                print("Sync ongoing, waiting for 30 seconds.")
            time.sleep(30)

    def similarity_search_by_vectors_with_scores_and_embeddings(
            self,
            embeddings: List[float],
            k: int = 5,
    ) -> List[Tuple[Document, List[float], float]]:
        # TODO add metadata filtering
        output = []
        for query_embedding in embeddings:
            documents = []
            results = self._search_embedding(embedding=query_embedding, k=k).nearest_neighbors.neighbors
            for result in results:
                metadata, embedding = {}, None
                for feature in result.entity_key_values.key_values.features:
                    if feature.name != self._vector_store.text_embedding_field:
                        metadata[feature.name] = feature.value.string_value
                    else:
                        embedding = feature.value.double_array_value.values
                documents.append(
                    [
                        Document(page_content=metadata[self._vector_store.content_field], metadata=metadata),
                        result.distance,
                        embedding
                    ]
                )
            output.append(documents)
        return output

    def _search_embedding(self, embedding, k=5):
        query = NearestNeighborQuery(
            embedding=NearestNeighborQuery.Embedding(value=embedding),
            neighbor_count=k
        )
        try:
            return self._search_client.search_nearest_entities(
                request=feature_online_store_service.SearchNearestEntitiesRequest(
                    feature_view=self._feature_view.gca_resource.name,
                    query=query,
                    return_full_entity=True,  # returning entities with metadata
                )
            )
        except ServiceUnavailable:
            raise ServiceUnavailable("The Feature Store service is not available. Did you sync your BQ data?")

    def _get_search_client(self) -> FeatureOnlineStoreServiceClient:
        endpoint = self._online_store.gca_resource.dedicated_serving_endpoint.public_endpoint_domain_name
        return FeatureOnlineStoreServiceClient(client_options={"api_endpoint": endpoint})

    def _create_online_store(self) -> FeatureOnlineStore:
        # Search for existing Online store
        stores_list = FeatureOnlineStore.list(project=self._vector_store.project_id,
                                              location=self._vector_store.location)
        for store in stores_list:
            if store.name == self.online_store_name:
                return store
        # Create it otherwise
        if self.online_store_type == "bigtable":

            online_store_config = feature_online_store_pb2.FeatureOnlineStore(
                bigtable=feature_online_store_pb2.FeatureOnlineStore.Bigtable(
                    auto_scaling=feature_online_store_pb2.FeatureOnlineStore.Bigtable.AutoScaling(
                        min_node_count=1, max_node_count=3, cpu_utilization_target=50
                    )
                ),
                embedding_management=feature_online_store_pb2.FeatureOnlineStore.EmbeddingManagement(
                    enabled=True
                ),
            )

            create_store_lro = self._admin_client.create_feature_online_store(
                feature_online_store_admin_service_pb2.CreateFeatureOnlineStoreRequest(
                    parent=f"projects/{self._vector_store.project_id}/locations/{self._vector_store.location}",
                    feature_online_store_id=self.online_store_name,
                    feature_online_store=online_store_config,
                )
            )
            print(create_store_lro.result())
        # TODO test optimised works
        elif self.online_store_type == "optimized":
            FeatureOnlineStore.create_optimized_store(
                name=self.online_store_name,
                project=self._vector_store.project_id,
                location=self.location
            )
        else:
            raise ValueError(f"{self.online_store_type} not allowed. Accepted values are `bigtable` or `optimized`.")

        stores_list = FeatureOnlineStore.list(project=self._vector_store.project_id,
                                              location=self._vector_store.location)
        for store in stores_list:
            if store.name == self.online_store_name:
                return store

    def _create_feature_view(self) -> FeatureView:
        # Search for existing Feature view
        fv_list = FeatureView.list(feature_online_store_id=self._online_store.gca_resource.name)
        for fv in fv_list:
            if fv.name == self.view_name:
                return fv
        # Create it otherwise
        big_query_source = FeatureViewBigQuerySource(
            uri=f"bq://{self._vector_store._full_table_id}", entity_id_columns=[self._vector_store.doc_id_field]
        )
        index_config = utils.IndexConfig(
            embedding_column=self._vector_store.text_embedding_field,
            crowding_column=None,
            filter_columns=None,
            dimensions=self._vector_store._embedding_dimension,
            distance_measure_type=utils.DistanceMeasureType.DOT_PRODUCT_DISTANCE,
            algorithm_config=utils.TreeAhConfig(),
        )

        return self._online_store.create_feature_view(
            name=self.view_name,
            source=big_query_source,
            sync_config=self.cron_schedule,
            index_config=index_config,
            project=self._vector_store.project_id,
            location=self._vector_store.location
        )


def validate_column_in_bq_schema(
        columns,
        column_name: str,
        expected_types: list,
        expected_modes: list,
):
    """Validates a column within a BigQuery schema.

    Args:
        columns: A dictionary of BigQuery SchemaField objects representing the table schema.
        column_name: The name of the column to validate.
        expected_types: A list of acceptable data types for the column.
        expected_modes: A list of acceptable modes for the column.

    Raises:
        ValueError: If the column doesn't exist, has an unacceptable type, or has an unacceptable mode.
    """

    if column_name not in columns:
        raise ValueError(f"Column {column_name} is missing from the schema.")

    column = columns[column_name]

    if column.field_type not in expected_types:
        raise ValueError(f"Column {column_name} must be one of the following types: {expected_types}")

    if column.mode not in expected_modes:
        raise ValueError(f"Column {column_name} must be one of the following modes: {expected_modes}")


class VertexFeatureStore(VectorStore, BaseModel):
    """Google Cloud Feature Store vector store.

    To use, you need the following packages installed:
        google-cloud-bigquery
    """
    embedding: Any
    project_id: str
    dataset_name: str
    table_name: str
    location: str
    executor: Union[BigQueryExecutor, FeatureOnlineStoreExecutor, BruteForceExecutor] = BigQueryExecutor()
    extra_fields: Dict[str, str] = None
    content_field: str = "content"
    text_embedding_field: str = "text_embedding"
    doc_id_field: str = "doc_id"
    credentials: Optional[Any] = None

    def model_post_init(self, __context):
        """Constructor for FeatureStore.
        """
        self._bq_client = bigquery.Client(
            project=self.project_id,
            location=self.location,
            credentials=self.credentials,
        )

        self._logger = logging.getLogger(__name__)
        self._embedding_dimension = len(self.embedding.embed_query("test"))
        self._full_table_id = (
            f"{self.project_id}." f"{self.dataset_name}." f"{self.table_name}"
        )
        self.executor.set_vector_store(vector_store=self)
        self._initialize_bq_table()
        self._validate_bq_table()
        self._logger.info(
            f"BigQuery table {self._full_table_id} initialized/validated as persistent storage. "
            f"Access via BigQuery console:\n https://console.cloud.google.com/bigquery?project={self.project_id}"
            f"&ws=!1m5!1m4!4m3!1s{self.project_id}!2s{self.dataset_name}!3s{self.table_name}"
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self.embedding

    @property
    def full_table_id(self) -> str:
        return self._full_table_id

    def _validate_bq_table(self):
        table_ref = bigquery.TableReference.from_string(self._full_table_id)

        try:
            table = self._bq_client.get_table(self.full_table_id)  # Attempt to retrieve the table information
        except NotFound:
            self._logger.debug(
                f"Couldn't find table {self.full_table_id}. Table will be created once documents are added")
            return

        table = self._bq_client.get_table(table_ref)
        schema = table.schema.copy()
        if schema:  ## Check if table has a schema
            columns = {c.name: c for c in schema}
            validate_column_in_bq_schema(column_name=self.doc_id_field, columns=columns,
                                         expected_types=["STRING"], expected_modes=["NULLABLE", "REQUIRED"])
            validate_column_in_bq_schema(column_name=self.content_field, columns=columns, expected_types=["STRING"],
                                         expected_modes=["NULLABLE", "REQUIRED"])
            validate_column_in_bq_schema(column_name=self.text_embedding_field, columns=columns,
                                         expected_types=["FLOAT", "FLOAT64"],
                                         expected_modes=["REPEATED"])
            if self.extra_fields is None:
                extra_fields = {}
                for column in schema:
                    if column.name not in [self.doc_id_field, self.content_field, self.text_embedding_field]:
                        # Check for unsupported REPEATED mode
                        if column.mode == "REPEATED":
                            raise ValueError(
                                f"Column '{column.name}' is REPEATED. REPEATED fields are not supported in this context.")
                        extra_fields[column.name] = column.field_type
                self.extra_fields = extra_fields
            else:
                for field, type in self.extra_fields.items():
                    validate_column_in_bq_schema(column_name=field, columns=columns,
                                                 expected_types=[type], expected_modes=["NULLABLE", "REQUIRED"])
            self._logger.debug(f"Table {self.full_table_id} validated")
        return table_ref

    def _initialize_bq_table(self) -> Any:
        """Validates or creates the BigQuery table."""
        self._bq_client.create_dataset(dataset=self.dataset_name, exists_ok=True)
        table_ref = bigquery.TableReference.from_string(self._full_table_id)
        self._bq_client.create_table(table_ref, exists_ok=True)
        return table_ref

    def sync(self):
        self.executor.sync()

    def add_texts(  # type: ignore[override]
            self,
            texts: List[str],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: List of strings to add to the vectorstore.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        embs = self.embedding.embed_documents(texts)
        return self.add_texts_with_embeddings(texts=texts, embs=embs, metadatas=metadatas, **kwargs)

    def add_texts_with_embeddings(
            self,
            texts: List[str],
            embs: List[List[float]],
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            ids: List of unique ids in string format
            texts: List of strings to add to the vectorstore.
            embs: List of lists of floats with text embeddings for texts.
            metadatas: Optional list of metadata associated with the texts.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        ids = [uuid.uuid4().hex for _ in texts]

        values_dict: List[Dict[str, List[Any]]] = []
        for idx, text, emb, metadata_dict in zip(ids, texts, embs, metadatas):
            record = {
                self.doc_id_field: idx,
                self.content_field: text,
                self.text_embedding_field: emb
            }
            values_dict.append(record | metadata_dict)

        table = self._bq_client.get_table(self.full_table_id)  # Attempt to retrieve the table information
        df = pd.DataFrame(values_dict)
        job = self._bq_client.load_table_from_dataframe(df, table)
        job.result()
        self._validate_bq_table()
        self._logger.debug(f"stored {len(ids)} records in BQ")
        self.sync()
        return ids

    def get_documents(
            self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search documents by their ids or metadata values.

        Args:
            ids: List of ids of documents to retrieve from the vectorstore.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
        Returns:
            List of ids from adding the texts into the vectorstore.
        """

        metadata_fields = list(self.extra_fields.keys())
        metadata_fields_str = ", ".join(metadata_fields)

        if ids and len(ids) > 0:

            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ArrayQueryParameter("ids", "STRING", ids),
                ]
            )
            id_expr = f"{self.doc_id_field} IN UNNEST(@ids)"
        else:
            job_config = None
            id_expr = "TRUE"
        if filter:
            filter_expressions = []
            for i in filter.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"`{metadata_fields_str}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = (
                        f"JSON_VALUE(`{metadata_fields_str}`,'$.{i[0]}')" f' = "{val}"'
                    )
                filter_expressions.append(expr)
            filter_expression_str = " AND ".join(filter_expressions)
            where_filter_expr = f" AND ({filter_expression_str})"
        else:
            where_filter_expr = ""

        job = self._bq_client.query(
            f"""
                    SELECT * FROM `{self.full_table_id}` WHERE {id_expr}
                    {where_filter_expr}
                    """,
            job_config=job_config,
        )
        docs: List[Document] = []
        for row in job:
            metadata = {}
            for field in self.extra_fields:
                metadata[field] = row[field]
            metadata["__id"] = row[self.doc_id_field]
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            docs.append(doc)
        return docs

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if not ids or len(ids) == 0:
            return True
        from google.cloud import bigquery  # type: ignore[attr-defined]

        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("ids", "STRING", ids),
            ]
        )
        self._bq_client.query(
            f"""
                    DELETE FROM `{self.full_table_id}` WHERE {self.doc_id_field}
                    IN UNNEST(@ids)
                    """,
            job_config=job_config,
        ).result()
        self.sync()
        return True

    async def adelete(
            self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def similarity_search_by_vectors(self, embeddings, k=5, with_scores=False, with_embeddings=False):
        """
        Core similarity search function. Handles a list of embedding vectors, optionally
        returning scores and embeddings.
        """

        results = self.executor.similarity_search_by_vectors_with_scores_and_embeddings(embeddings, k)

        # Process results based on options
        for i, query_results in enumerate(results):
            if not with_scores and not with_embeddings:
                # return only docs
                results[i] = [x[0] for x in query_results]
            elif not with_embeddings:
                # return only docs and score
                results[i] = [[x[0], x[1]] for x in query_results]
            elif not with_scores:
                # return only docs and embeddings
                results[i] = [[x[0], x[2]] for x in query_results]

        return results

    def similarity_search_by_vector(
            self, embedding: List[float], k: int = 5) -> List[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.

        Returns:
            List of Documents most similar to the query vector.
        """
        return self.similarity_search_by_vectors(embeddings=[embedding], k=k)[0]

    def similarity_search_by_vector_with_score(self, embedding, k=5):
        """
        Searches for similar items given a single embedding vector and returns results with scores.
        """
        return self.similarity_search_by_vectors([embedding], k, with_scores=True)[0]

    def similarity_search(self, query, k=5):
        """
        Searches for similar items given a query string and returns results without scores or embeddings.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vectors([embedding], k)[0]

    def similarity_search_with_score(self, query, k=5):
        """
        Searches for similar items given a query string and returns results with scores.
        """
        embedding = self.embedding.embed_query(query)
        return self.similarity_search_by_vector_with_score(embedding, k)

    def batch_search(self, embeddings=None, queries=None, k=5, with_scores=False, with_embeddings=False):
        """
        Multi-purpose batch search function. Accepts embeddings, queries, or both.
        """
        if embeddings is None and queries is None:
            raise ValueError("At least one of 'embeddings' or 'queries' must be provided.")

        if embeddings is not None and queries is not None:
            raise ValueError("Only one parameter between 'embeddings' or 'queries' must be provided")

        if embeddings is not None and not isinstance(embeddings[0], list):
            embeddings = [embeddings]  # Ensure embeddings is a list of lists

        if queries is not None:
            embeddings = self.embedding.embed_documents(queries, task="RETRIEVAL_QUERY")

        return self.similarity_search_by_vectors(embeddings, k, with_scores, with_embeddings)

    def max_marginal_relevance_search(
            self,
            query: str,
            k: int = 5,
            fetch_k: int = 25,
            lambda_mult: float = 0.5,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: search query text.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult
        )

    def max_marginal_relevance_search_by_vector(
            self,
            embedding: List[float],
            k: int = 5,
            fetch_k: int = 25,
            lambda_mult: float = 0.5
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter on metadata properties, e.g.
                            {
                                "str_property": "foo",
                                "int_property": 123
                            }
            brute_force: Whether to use brute force search. Defaults to False.
            fraction_lists_to_search: Optional percentage of lists to search,
                must be in range 0.0 and 1.0, exclusive.
                If Node, uses service's default which is 0.05.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        doc_tuples = \
            self.similarity_search_by_vectors(embeddings=[embedding], k=fetch_k, with_embeddings=True,
                                              with_scores=True)[0]
        doc_embeddings = [d[2] for d in doc_tuples]
        mmr_doc_indexes = maximal_marginal_relevance(
            np.array(embedding), doc_embeddings, lambda_mult=lambda_mult, k=k
        )
        return [doc_tuples[i][0] for i in mmr_doc_indexes]

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ):
        raise NotImplementedError

    @validate_call
    def set_executor(self, executor: Union[BigQueryExecutor, FeatureOnlineStoreExecutor, BruteForceExecutor]):
        self.executor = executor
        self.executor.set_vector_store(vector_store=self)

    def as_retriever(self, **kwargs: Any) -> VertexFeatureStoreRetriever:
        tags = kwargs.pop("tags", None) or []
        tags.extend(self._get_retriever_tags())
        return VertexFeatureStoreRetriever(vectorstore=self, **kwargs)


class VertexFeatureStoreRetriever(VectorStoreRetriever):
    vectorstore: VertexFeatureStore
    """VectorStore to use for retrieval."""

    def batch(
            self,
            inputs: List[Input],
            k=5,
    ) -> List[Output]:
        return self.vectorstore.search_by_queries(inputs, k=k)
