"""CLI del sistema RAG."""
import os
import sys
from src.config import get_settings, Settings
from src.exceptions import RAGError, ConfigurationError, VectorStoreNotInitializedError
from src.logger import setup_logger, get_logger
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_system import RAGSystem
from src.embeddings import LocalEmbeddings

logger = get_logger("rag.cli")

def print_banner():
    print("""
    ======================================
           SISTEMA RAG LOCAL
    ======================================
    """)

def print_menu():
    print("""
    1. Indexar documentos
    2. Cargar índice existente
    3. Consultar
    4. Eliminar índice
    5. Salir
    """)

def index_documents(rag: RAGSystem) -> bool:
    """Maneja la indexación desde la UI."""
    print("\n--- Indexación ---")
    print("1. Carpeta 'data/documentos'")
    print("2. Archivo específico")
    
    opt = input("Opción: ").strip()
    path = "./data/documentos" if opt == "1" else input("Ruta: ").strip() if opt == "2" else None

    if not path: return False
    
    if not os.path.exists(path):
        print(f"[!] No existe: {path}")
        return False

    try:
        return rag.index_documents(path)
    except RAGError as e:
        logger.error(e)
        print(f"[ERROR] {e}")
        return False

def query_loop(rag: RAGSystem):
    """Bucle de consultas."""
    print("\n--- Consulta (Escribe 'salir' para terminar) ---")
    while True:
        q = input("\n[?] Pregunta: ").strip()
        if q.lower() in ["salir", "exit"]: break
        if not q: continue

        try:
            res = rag.query(q)
            print(f"\n[R] {res.answer}\n")
            if src := res.format_sources(): print(src)
        except Exception as e:
            print(f"[ERROR] {e}")

def main():
    setup_logger("rag", "INFO")
    print_banner()

    try:
        settings = get_settings()
        # Inyección de dependencias explicita
        rag = RAGSystem(
            settings=settings,
            vector_store=VectorStore(settings.vectorstore_path, LocalEmbeddings(settings.embedding_model_name)),
            document_loader=DocumentLoader(settings.chunk_size, settings.chunk_overlap)
        )
        
        index_loaded = False

        while True:
            print_menu()
            opt = input("Selección: ").strip()

            if opt == "1":
                if index_documents(rag): index_loaded = True
            elif opt == "2":
                if rag.load_existing_index():
                    index_loaded = True
                    print("[OK] Índice cargado")
                else:
                    print("[!] No hay índice guardado")
            elif opt == "3":
                if index_loaded or rag.vector_store.is_initialized():
                    query_loop(rag)
                else:
                    print("[!] Carga o indexa documentos primero")
            elif opt == "4":
                if input("¿Seguro? (s/n): ").lower() == "s":
                    rag.delete_index()
                    index_loaded = False
                    print("[OK] Eliminado")
            elif opt == "5":
                sys.exit(0)

    except ConfigurationError as e:
        print(f"[FATAL] Configuración: {e}")
        print("Verifica que Ollama esté corriendo y el modelo instalado.")
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    main()
