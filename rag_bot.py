import cohere
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Hardcoded Cohere API Key — make sure it's valid
cohere_api_key = "0LORZdUQhwRj0y3SMsr1xMVjqmtw3BcgVz0rUcBr"

# Initialize Cohere Client
co = cohere.Client(cohere_api_key)

chat_history = []

def get_most_relevant_chunks(query, chunks, top_k=3):
    """
    Find the most relevant text chunks for the given query using cosine similarity.
    """
    vectorizer = TfidfVectorizer().fit_transform(chunks + [query])
    cosine_similarities = cosine_similarity(vectorizer[-1], vectorizer[:-1])
    top_indices = cosine_similarities[0].argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def ask_cohere(query, context):
    """
    Ask Cohere's Chat API using the 'command-r' model with context.
    """
    try:
        response = co.chat(
            model='command-r',
            message=query,
            documents=[{"text": context}],
            temperature=0.5
        )
        return response.text.strip()
    except Exception as e:
        print(f"Error from Cohere: {e}")
        return "An error occurred while processing your request. Please check the API and try again."

def answer_query(query, chunks):
    """
    Given a query, find the most relevant chunks and ask Cohere for an answer.
    """
    relevant_chunks = get_most_relevant_chunks(query, chunks)
    context = "\n\n".join(relevant_chunks)
    answer = ask_cohere(query, context)
    chat_history.append({
        "question": query,
        "answer": answer,
        "source": relevant_chunks
    })
    return answer, relevant_chunks

# Example usage
if __name__ == "__main__":
    chunks = [
        "Cohere is an AI platform that offers powerful large language models for developers.",
        "The generate endpoint can be used to create answers, summaries, or completions.",
        "The platform supports fine-tuning and custom dataset ingestion for better results.",
        "Cohere's models can be used for retrieval-augmented generation tasks.",
        "The chat API supports document grounding and dynamic context injection."
    ]
    
    query = "How can Cohere help with summarizing documents?"
    answer, sources = answer_query(query, chunks)

    print("\nAnswer:\n", answer)
    print("\nSource chunks used:")
    for i, src in enumerate(sources, 1):
        print(f"{i}. {src}")
