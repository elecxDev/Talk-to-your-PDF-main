# Import necessary libraries
import streamlit as st
import numpy as np
import os, requests, tempfile, json
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import ProgrammingError
from pdfminer.high_level import extract_text as pdf_extract_text
from streamlit_lottie import st_lottie_spinner



# Function to load Lottie animations using URL
@st.cache_data
def load_lottieurl(url):
    """
    Fetches and caches a Lottie animation from a provided URL.

    Args:
    url (str): The URL of the Lottie animation.

    Returns:
    dict: The Lottie animation JSON or None if the request fails.
    """
    r = requests.get(url)  # Perform the GET request
    if r.status_code != 200:
        return None  # Return None if request failed
    return r.json()  # Return the JSON content of the Lottie animation


# Load a specific Lottie animation to be used in the app
loading_animation = load_lottieurl(
    "https://lottie.host/5ac92c74-1a02-40ff-ac96-947c14236db1/u4nCMW6fXU.json"
)


# Class for processing uploaded PDFs before user interaction
class PreRunProcessor:
    """
    Processes uploaded PDF files by extracting text and generating embeddings.
    """

    def __init__(self):
        """
        Initializes the processor with database connection and Mistral client.
        """
        # Establish connection to the PostgreSQL database from the Supabase platform
        self.engine = create_engine(
            st.secrets["SUPABASE_POSTGRES_URL"], echo=True, client_encoding="utf8"
        )
        # Create a session maker bound to this engine
        self.Session = sessionmaker(bind=self.engine)
        self.ensure_table_exists()
        self.mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

    def pdf_to_text(self, uploaded_file, chunk_length: int = 1000) -> list:
        """
        Extracts text from the uploaded PDF and splits it into manageable chunks.

        Args:
        uploaded_file (UploadedFile): The PDF file uploaded by the user.
        chunk_length (int): The desired length of each text chunk.

        Returns:
        list: A list of text chunks ready for embedding generation.
        """
        # Extract text from the uploaded PDF
        text = pdf_extract_text(uploaded_file)
        # Split the text into chunks
        chunks = [
            text[i : i + chunk_length].replace("\n", "")
            for i in range(0, len(text), chunk_length)
        ]
        return self._generate_embeddings(chunks)

    def define_vector_store(self, embeddings: list) -> bool:
        """
        Stores the generated embeddings in the database.

        Args:
        embeddings (list): A list of dictionaries containing text and their corresponding embeddings.

        Returns:
        bool: True if the operation succeeds, False otherwise.
        """
        session = self.Session()  # Create a new database session
        try:
            # Truncate the existing table and insert new embeddings
            session.execute(text("TRUNCATE TABLE pdf_holder RESTART IDENTITY CASCADE;"))
            for embedding in embeddings:
                # Insert each embedding into the pdf_holder table
                session.execute(
                    text(
                        "INSERT INTO pdf_holder (text, embedding) VALUES (:text, :embedding)"
                    ),
                    {"text": embedding["text"], "embedding": embedding["vector"]},
                )
            session.commit()  # Commit the changes
            return True
        except ProgrammingError as e:
            if 'relation "pdf_holder" does not exist' in str(e.orig.pgerror):
                # If the table doesn't exist, create it and the necessary extension
                session.rollback()
                session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                session.execute(
                    text("""
                    CREATE TABLE pdf_holder (
                        id SERIAL PRIMARY KEY,
                        text TEXT,
                        embedding VECTOR(1024)
                    );
                """)
                )
                session.commit()
                return False
            else:
                raise
        finally:
            session.close()  # Close the session

    def _generate_embeddings(self, chunks: list) -> list:
        """
        Generates embeddings for each text chunk using Mistral.

        Args:
        chunks (list): A list of text chunks.

        Returns:
        list: A list of dictionaries containing text chunks and their corresponding embeddings.
        """
        try:
            # Filter out null characters from each chunk
            cleaned_chunks = [chunk.replace("\x00", "") for chunk in chunks]
            embeddings = []
            
            for chunk in cleaned_chunks:
                response = self.mistral_client.embeddings(
                    model="mistral-embed",
                    input=[chunk]
                )
                embedding = response.data[0].embedding
                embeddings.append({
                    "vector": embedding,
                    "text": chunk
                })
                    
            return embeddings
        except Exception as e:
            st.error(f"An error occurred during embeddings generation: {e}")
            return []

    def ensure_table_exists(self):
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(
                text("""
                CREATE TABLE IF NOT EXISTS pdf_holder (
                    id SERIAL PRIMARY KEY,
                    text TEXT,
                    embedding VECTOR(1024)
                );
            """)
            )
            conn.commit()


# Function to process the uploaded PDF before any user interaction
def process_pre_run(uploaded_file):
    processor_class = PreRunProcessor()
    try:
        embeddings = processor_class.pdf_to_text(uploaded_file)
        if not embeddings:
            st.error("Failed to generate embeddings from the PDF.")
            return
        if not processor_class.define_vector_store(embeddings):
            st.error("Failed to store the PDF embedding.")
        else:
            st.success("PDF successfully uploaded and processed.")
    except Exception as e:
        st.error(f"An error occurred during pre-processing: {e}")


##### Intent services #####


class IntentService:
    """
    Handles the detection of malicious intent in user queries, conversion of questions to embeddings,
    and checks the relatedness of questions to PDF content via database queries.
    """

    def __init__(self):
        """
        Initializes the IntentService with database connection and Mistral client.
        """
        # Establish a connection to the PostgreSQL database hosted on the Supabase platform
        self.engine = create_engine(
            st.secrets["SUPABASE_POSTGRES_URL"], echo=True, client_encoding="utf8"
        )
        self.mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

    def detect_malicious_intent(self, question):
        """
        Simple keyword-based malicious intent detection (free alternative).

        Args:
            question (str): The user's question as a string.

        Returns:
            tuple: A boolean indicating if the question was flagged and a message explaining the result.
        """
        try:
            # Simple keyword filtering for malicious content
            malicious_keywords = ['hack', 'attack', 'exploit', 'malware', 'virus', 'illegal', 'bomb', 'weapon']
            question_lower = question.lower()
            
            is_flagged = any(keyword in question_lower for keyword in malicious_keywords)
            
            if is_flagged:
                return True, "This question has been flagged for potentially inappropriate content..."
            else:
                return False, "No malicious intent detected..."
        except Exception as e:
            return None, f"Error in moderation: {str(e)}"

    def query_database(self, query):
        """
        Executes a SQL query on the connected PostgreSQL database and returns the first result.

        Args:
            query (str): SQL query string to be executed.

        Returns:
            sqlalchemy.engine.row.RowProxy or None: The first result row of the query or None if no results.
        """
        # Connect to the database and execute the given query
        with self.engine.connect() as connection:
            result = connection.execute(text(query)).fetchone()
            # Return the result if available; otherwise, return None
            return result if result else None

    def question_to_embeddings(self, question):
        """
        Converts a user's question into vector embeddings using Mistral.

        Args:
            question (str): The user's question as a string.

        Returns:
            list: The vectorized form of the question as a list or an empty list on failure.
        """
        try:
            response = self.mistral_client.embeddings(
                model="mistral-embed",
                input=[question]
            )
            embedding = response.data[0].embedding
            # Verify the dimensionality of the embedding
            if len(embedding) != 1024:
                raise ValueError(
                    "The dimensionality of the question embedding does not match the expected 1024 dimensions."
                )
            return embedding
        except Exception as e:
            print(f"Error embedding the question: {e}")
            return []

    def check_relatedness_to_pdf_content(self, question):
        """
        Determines if a user's question is related to PDF content stored in the database by querying for similar embeddings.

        Args:
            question (str): The user's question as a string.

        Returns:
            tuple: A boolean indicating relatedness and a message explaining the result.
        """
        # Convert the question to vector embeddings
        question_vectorized = self.question_to_embeddings(question)

        try:
            # Query the database for the closest embedding to the question's embedding
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                    SELECT id, text, embedding <=> CAST(:question_vector AS VECTOR) AS distance 
                    FROM pdf_holder
                    ORDER BY distance ASC
                    LIMIT 1;
                """),
                    {"question_vector": question_vectorized},
                ).fetchone()

                if result:
                    # Determine if the closest embedding is below a certain threshold
                    _, _, distance = result
                    threshold = 0.65  # Define a threshold for relatedness
                    if distance < threshold:
                        # Return true and a message if the question is related to the PDF content
                        return True, "Question is related to the PDF content..."
                    else:
                        # Return false and a message if the question is not sufficiently related
                        return False, "Question is not related to the PDF content..."
                else:
                    # Return false and a message if no embedding was found in the database
                    return False, "No match found in the database."
        except Exception as e:
            # Log and return false in case of an error during the database query
            print(f"Error searching the database: {e}")
            return False, f"Error searching the database: {e}"


##### Information retrieval service #####


class InformationRetrievalService:
    """
    Provides services for searching vectorized questions within a vector store in the database.
    """

    def __init__(self):
        """
        Initializes the InformationRetrievalService with database connection and Mistral client.
        """
        # Establish connection to the PostgreSQL database on the Supabase platform
        self.engine = create_engine(
            st.secrets["SUPABASE_POSTGRES_URL"], echo=True, client_encoding="utf8"
        )
        # Create a session maker bound to this engine
        self.Session = sessionmaker(bind=self.engine)
        self.mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

    def search_in_vector_store(self, question_embedding, k: int = 1) -> str:
        """
        Searches for the closest matching text in the vector store using a pre-computed embedding.

        Args:
            question_embedding: The pre-computed embedding vector for the question.
            k (int): The number of top results to retrieve, defaults to 1.

        Returns:
            str: The text of the closest matching document or None if no match is found.
        """
        try:
            # SQL query to find the closest match in the vector store
            sql_query = text("""
                SELECT id, text, embedding <=> CAST(:query_vector AS VECTOR) AS distance
                FROM pdf_holder
                ORDER BY distance
                LIMIT :k
            """)
            # Execute the query with the vectorized question and k value
            with self.engine.connect() as conn:
                results = conn.execute(
                    sql_query, {"query_vector": question_embedding, "k": k}
                ).fetchall()
                if results:
                    # Return the text of the closest match if results are found
                    return results[0].text
                else:
                    # Display an error if no matching documents are found
                    st.error("No matching documents found.")
        except Exception as e:
            st.error(f"Error searching vector store: {e}")
            return None


##### Response service #####
class ResponseService:
    """
    Generates responses to user questions by integrating with Ollama.
    """

    def __init__(self):
        """
        Initializes the ResponseService with Mistral client.
        """
        self.mistral_client = MistralClient(api_key=st.secrets["MISTRAL_API_KEY"])

    def generate_response(self, question, retrieved_info):
        """
        Generates a response using Mistral based on the provided question and retrieved information.

        Args:
            question (str): The user's question.
            retrieved_info (str): Information retrieved that is related to the question.

        Returns:
            str: The generated response or an error message if no response is available.
        """
        try:
            messages = [
                ChatMessage(role="user", content=f"""Answer the user's question using the PDF context. Format your response in markdown for better readability.

PDF Context: {retrieved_info}

Question: {question}

Instructions:
- Format response in markdown with proper headings, bullet points, etc.
- Keep it concise but well-structured
- Use **bold** for key terms
- Use bullet points for lists
- Use ### for subheadings if needed
- Only answer what was specifically asked
- If you can't answer from context, say so briefly""")
            ]
            
            response = self.mistral_client.chat(
                model="mistral-small",
                messages=messages
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating response: {e}")
            return "Sorry, I couldn't generate a response at this time."


###### Independant & dependant of the function's class ######





# Orchestrates the processing of user questions regarding PDF content
def intent_orchestrator(service_class, user_question):
    """
    Orchestrates the process of checking a user's question for malicious intent and relevance to PDF content.

    Args:
        service_class: The class instance providing the services for intent detection and content relevance.
        user_question: The question posed by the user.

    Returns:
        A tuple containing the vectorized question and the original question if relevant, or (None, None) otherwise.
    """
    # Detect malicious intent in the user's question
    is_flagged, flag_message = service_class.detect_malicious_intent(user_question)
    st.write(flag_message)  # Display the flag message

    if is_flagged:
        # If the question is flagged, do not process further
        st.error("Your question was not processed. Please try a different question.")
        return (None, None)

    # Check if the question is related to the PDF content
    related, relatedness_message = service_class.check_relatedness_to_pdf_content(
        user_question
    )
    st.write(relatedness_message)  # Display the relatedness message

    if related:
        # If the question is related, proceed with processing
        vectorized_question = service_class.question_to_embeddings(user_question)
        st.success(
            "Your question was processed successfully. Now fetching an answer..."
        )
        return (vectorized_question, user_question)
    else:
        # If not related, do not process further
        st.error("Your question was not processed. Please try a different question.")
        return (None, None)


# Starts the question processing workflow
def process_user_question(service_class, user_question):
    """
    Initiates the processing of a user's question through various services.

    Args:
        service_class: The class instance providing services for processing the user's question.
        user_question: The question posed by the user.

    Returns:
        The result of the intent orchestration process.
    """
    # Orchestrates the intent processing of the user's question
    result = intent_orchestrator(service_class, user_question)
    return result


# Initiates the retrieval process for information related to the user's question
def process_retrieval(question: str) -> str:
    """
    Retrieves information related to the question from the vector store.

    Args:
        question (str): The user's question.

    Returns:
        Retrieved information related to the user's question.
    """
    service = InformationRetrievalService()
    retrieved_info = service.search_in_vector_store(question)
    return retrieved_info


# Generates a response based on the user's question and the retrieved information
def process_response(retrieved_info, question):
    """
    Generates a response to the user's question based on retrieved information.

    Args:
        retrieved_info: Information related to the user's question retrieved from the vector store.
        question: The original question posed by the user.

    Returns:
        A generated response to the user's question.
    """
    response_service_processor = ResponseService()
    final_response = response_service_processor.generate_response(
        question, retrieved_info
    )
    return final_response


def main():
    # Check API key
    if "MISTRAL_API_KEY" not in st.secrets:
        st.error("Mistral API key not found!")
        return

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "service_class" not in st.session_state:
        st.session_state.service_class = None

    # Dark gray theme with proper layout
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        .main {
            font-family: 'Inter', sans-serif;
        }
        
        /* Message styling */
        .user-message {
            background: #4a5568;
            color: white;
            padding: 15px 20px;
            border-radius: 20px 20px 5px 20px;
            margin: 10px 0 10px 50px;
            box-shadow: 0 4px 15px rgba(74, 85, 104, 0.3);
        }
        
        .ai-message {
            background: #2d3748;
            color: #e2e8f0;
            padding: 15px 20px;
            border-radius: 20px 20px 20px 5px;
            margin: 10px 50px 10px 0;
            border: 1px solid #4a5568;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Upload area */
        .upload-container {
            background: #2d3748;
            border: 2px dashed #4a5568;
            border-radius: 16px;
            padding: 30px;
            margin: 20px 0;
            text-align: center;
        }
        
        /* Input styling */
        .stTextInput>div>div>input {
            background: #2d3748 !important;
            border: 2px solid #4a5568 !important;
            border-radius: 12px !important;
            padding: 12px 16px !important;
            color: white !important;
            font-size: 16px !important;
        }
        
        .stTextInput>div>div>input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 24px !important;
            font-weight: 500 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4) !important;
        }
        
        /* File uploader */
        .stFileUploader {
            border: 2px dashed #4a5568;
            border-radius: 16px;
            padding: 24px;
            background: #2d3748;
        }
        
        /* Success/Error messages */
        .stSuccess {
            background: rgba(72, 187, 120, 0.1) !important;
            border-left: 4px solid #48bb78 !important;
            border-radius: 8px !important;
            color: #68d391 !important;
        }
        
        .stError {
            background: rgba(245, 101, 101, 0.1) !important;
            border-left: 4px solid #f56565 !important;
            border-radius: 8px !important;
            color: #fc8181 !important;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # Title
    if not st.session_state.pdf_processed:
        st.markdown("""
            <div style="text-align: center; padding: 2rem 0;">
                <h1 style="color: white; font-size: 3rem; font-weight: 700; margin-bottom: 0.5rem;">
                    Talk to your PDF
                </h1>
                <p style="color: #a0aec0; font-size: 1.2rem; margin: 0;">Upload a PDF and ask questions about its content</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align: center; padding: 10px 0;"><h2 style="color: white; margin: 0;">PDF Chat</h2></div>', unsafe_allow_html=True)

    # PDF Upload
    if not st.session_state.pdf_processed:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: white; margin-bottom: 20px;">Upload your PDF document</h3>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file to analyze", type=["pdf"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            with st.spinner("Processing your PDF..."):
                process_pre_run(uploaded_file)
                st.session_state.pdf_processed = True
                st.session_state.service_class = IntentService()
                st.session_state.messages.append({
                    "role": "system", 
                    "content": f"PDF '{uploaded_file.name}' processed successfully! Ask questions about its content."
                })
                st.rerun()
    
    # Chat Interface
    if st.session_state.pdf_processed:
        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">You: {message["content"]}</div>', unsafe_allow_html=True)
            elif message["role"] == "assistant":
                st.markdown('<div class="ai-message">AI:</div>', unsafe_allow_html=True)
                st.markdown(message["content"])
            elif message["role"] == "system":
                st.success(message["content"])
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Ask a question about the PDF:", placeholder="What is this document about?")
            submit_button = st.form_submit_button("Send Message")
        
        # Process user input
        if submit_button and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st_lottie_spinner(loading_animation, quality="high", height="100px", width="100px"):
                result = process_user_question(st.session_state.service_class, user_input)
                
                if result[0] is not None:
                    vectorized_question, question = result
                    retrieved_info = process_retrieval(vectorized_question)
                    final_response = process_response(retrieved_info, question)
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                else:
                    st.session_state.messages.append({"role": "assistant", "content": "I couldn't process that question. Please try asking something related to the PDF content."})
            
            st.rerun()
        
        # Reset options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload New PDF"):
                st.session_state.pdf_processed = False
                st.session_state.messages = []
                st.session_state.service_class = None
                st.rerun()
        with col2:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()


# Entry point of the Streamlit app
if __name__ == "__main__":
    main()
