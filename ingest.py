"""one time Ingestion: load → chunk → deduplicate → embed → index."""

from src.ingestion.ingestion_pipeline import IngestionPipeline


pipeline = IngestionPipeline()
pipeline.ingest_directory("Research_papers/")

        
print("Ingestion is done")