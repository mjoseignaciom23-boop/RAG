"""Script de prueba de integracion del sistema RAG."""
import os
import sys

# Agregar el directorio al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import get_settings
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_system import RAGSystem

def main():
    print("=" * 60)
    print("TEST DE INTEGRACION DEL SISTEMA RAG")
    print("=" * 60)

    # 1. Cargar configuracion
    print("\n[1/4] Cargando configuracion...")
    settings = get_settings()
    print(f"  - Modelo: {settings.llm_model_name}")
    print(f"  - Chunk size: {settings.chunk_size}")
    print("  OK!")

    # 2. Crear componentes
    print("\n[2/4] Creando componentes del sistema...")
    document_loader = DocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    vector_store = VectorStore(
        persist_directory="./test_vectorstore_temp",
    )

    rag_system = RAGSystem(
        settings=settings,
        vector_store=vector_store,
        document_loader=document_loader,
    )
    print("  OK!")

    # 3. Indexar documento de prueba
    print("\n[3/4] Indexando documentos de prueba...")
    success = rag_system.index_documents("./data/documentos")
    if success:
        print("  OK!")
    else:
        print("  ERROR: No se pudieron indexar los documentos")
        return

    # 4. Hacer una consulta
    print("\n[4/4] Realizando consulta de prueba...")
    question = "Que es el sistema RAG y cuales son sus componentes?"
    print(f"  Pregunta: {question}")

    result = rag_system.query(question)

    print("\n" + "=" * 60)
    print("RESPUESTA:")
    print("=" * 60)
    print(result.answer)

    print("\n" + "=" * 60)
    print("FUENTES:")
    print("=" * 60)
    print(result.format_sources())

    # Limpiar
    print("\n[OK] Limpiando recursos de prueba...")
    rag_system.delete_index()
    print("  OK!")

    print("\n" + "=" * 60)
    print("TEST COMPLETADO EXITOSAMENTE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
