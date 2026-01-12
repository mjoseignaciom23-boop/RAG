"""
CLI interactiva para el sistema RAG.
"""
import os
import sys

from src.config import get_settings, Settings
from src.exceptions import (
    RAGError,
    ConfigurationError,
    VectorStoreNotInitializedError,
)
from src.logger import setup_logger, get_logger
from src.document_loader import DocumentLoader
from src.vector_store import VectorStore
from src.rag_system import RAGSystem
from src.models import QueryResult

logger = get_logger("rag.cli")


def print_banner() -> None:
    """Imprime el banner de bienvenida."""
    banner = """
    =====================================================
    |                                                   |
    |         SISTEMA RAG - RETRIEVAL AUGMENTED        |
    |                   GENERATION                      |
    |                                                   |
    =====================================================
    """
    print(banner)


def print_menu() -> None:
    """Imprime el menú principal."""
    menu = """
    -----------------------------------------------------
    |  MENU PRINCIPAL                                   |
    -----------------------------------------------------
    |  1. Indexar documentos                            |
    |  2. Cargar indice existente                       |
    |  3. Hacer una consulta                            |
    |  4. Eliminar indice                               |
    |  5. Salir                                         |
    -----------------------------------------------------
    """
    print(menu)


def print_sources(result: QueryResult) -> None:
    """
    Imprime las fuentes de información.

    Args:
        result: Resultado de la consulta
    """
    formatted = result.format_sources()
    if formatted:
        print(formatted)


def index_documents(rag_system: RAGSystem) -> bool | None:
    """
    Menú para indexar documentos.

    Args:
        rag_system: Instancia del sistema RAG

    Returns:
        True si se indexó correctamente, None si se canceló
    """
    print("\n" + "=" * 60)
    print("INDEXAR DOCUMENTOS")
    print("=" * 60)
    print("\nOpciones:")
    print("1. Indexar todos los documentos de la carpeta 'data/documentos'")
    print("2. Indexar un archivo especifico")
    print("3. Volver al menu principal")

    choice = input("\nSelecciona una opcion (1-3): ").strip()

    if choice == "1":
        default_path = "./data/documentos"
        if os.path.exists(default_path):
            try:
                return rag_system.index_documents(default_path)
            except RAGError as e:
                logger.error(f"Error indexando: {e}")
                print(f"\n[ERROR] {e}")
                return None
        else:
            print(f"\n[ERROR] No existe el directorio: {default_path}")
            print("Por favor, crea la carpeta y anade documentos.")
            return None

    elif choice == "2":
        file_path = input("\nIngresa la ruta del archivo: ").strip()
        if os.path.exists(file_path):
            try:
                return rag_system.index_documents(file_path)
            except RAGError as e:
                logger.error(f"Error indexando: {e}")
                print(f"\n[ERROR] {e}")
                return None
        else:
            print(f"\n[ERROR] No existe el archivo: {file_path}")
            return None

    elif choice == "3":
        return None

    else:
        print("\n[ERROR] Opcion no valida")
        return None


def query_documents(rag_system: RAGSystem) -> None:
    """
    Menú para realizar consultas.

    Args:
        rag_system: Instancia del sistema RAG
    """
    print("\n" + "=" * 60)
    print("MODO CONSULTA")
    print("=" * 60)
    print("Escribe 'salir' para volver al menu principal")
    print("=" * 60)

    while True:
        question = input("\n[?] Tu pregunta: ").strip()

        if question.lower() in ["salir", "exit", "quit"]:
            break

        if not question:
            print("[!] Por favor, escribe una pregunta")
            continue

        try:
            result = rag_system.query(question)

            print("\n" + "=" * 60)
            print("[OK] RESPUESTA")
            print("=" * 60)
            print(f"\n{result.answer}")

            print_sources(result)

        except VectorStoreNotInitializedError as e:
            print(f"\n[ERROR] {e}")
            break
        except RAGError as e:
            logger.error(f"Error en consulta: {e}")
            print(f"\n[ERROR] {e}")


def create_rag_system(settings: Settings) -> RAGSystem:
    """
    Crea e inicializa el sistema RAG con todas sus dependencias.

    Args:
        settings: Configuración de la aplicación

    Returns:
        RAGSystem configurado
    """
    document_loader = DocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    vector_store = VectorStore(
        persist_directory=settings.vectorstore_path,
    )

    rag_system = RAGSystem(
        settings=settings,
        vector_store=vector_store,
        document_loader=document_loader,
    )

    return rag_system


def main() -> None:
    """Función principal."""
    setup_logger("rag", level="INFO")
    print_banner()

    try:
        settings = get_settings()
        rag_system = create_rag_system(settings)
        index_loaded = False

        while True:
            print_menu()
            choice = input("Selecciona una opcion (1-5): ").strip()

            if choice == "1":
                success = index_documents(rag_system)
                if success:
                    index_loaded = True

            elif choice == "2":
                try:
                    if rag_system.load_existing_index():
                        index_loaded = True
                        print("\n[OK] Indice cargado correctamente")
                    else:
                        print(
                            "\n[!] No se pudo cargar el indice. "
                            "Necesitas indexar documentos primero."
                        )
                except RAGError as e:
                    logger.error(f"Error cargando indice: {e}")
                    print(f"\n[ERROR] {e}")

            elif choice == "3":
                if not index_loaded and not rag_system.vector_store.is_initialized():
                    print(
                        "\n[!] Primero debes indexar documentos o cargar un indice existente"
                    )
                    print("   (Opciones 1 o 2 del menu)")
                else:
                    query_documents(rag_system)

            elif choice == "4":
                confirm = input(
                    "\nEstas seguro de eliminar el indice? (s/n): "
                ).strip().lower()
                if confirm == "s":
                    rag_system.delete_index()
                    index_loaded = False
                    print("\n[OK] Indice eliminado")
                else:
                    print("Operacion cancelada")

            elif choice == "5":
                print("\nHasta luego!")
                sys.exit(0)

            else:
                print("\n[ERROR] Opcion no valida. Por favor, selecciona 1-5")

    except ConfigurationError as e:
        logger.error(f"Error de configuracion: {e}")
        print(f"\n[ERROR] Error de configuracion: {e}")
        print("\nAsegurate de:")
        print("1. Tener el archivo .env configurado con tu OPENAI_API_KEY")
        print(
            "2. Haber instalado todas las dependencias (pip install -r requirements.txt)"
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nHasta luego!")
        sys.exit(0)

    except Exception as e:
        logger.exception(f"Error inesperado: {e}")
        print(f"\n[ERROR] Error inesperado: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
