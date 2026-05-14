import os
import networkx as nx
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

client = Client(
    host='https://ollama.com',
    headers={'Authorization': f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
)

def build_knowledge_graph():
    """
    Builds an in-memory knowledge graph of entities and relationships.
    In production, this data would live in a graph database like Neo4j.
    """
    print("🕸️ Building Knowledge Graph...")
    G = nx.Graph()

    # Add Nodes (Entities)
    movies = ["Inception", "Interstellar", "The Dark Knight", "Oppenheimer", "Tenet"]
    directors = ["Christopher Nolan"]
    actors = ["Leonardo DiCaprio", "Matthew McConaughey", "Christian Bale", "Cillian Murphy"]
    genres = ["Sci-Fi", "Action", "Drama", "Thriller"]

    for movie in movies: G.add_node(movie, type="Movie")
    for director in directors: G.add_node(director, type="Director")
    for actor in actors: G.add_node(actor, type="Actor")
    for genre in genres: G.add_node(genre, type="Genre")

    # Add Edges (Relationships)
    relationships = [
        ("Christopher Nolan", "DIRECTED", "Inception"),
        ("Christopher Nolan", "DIRECTED", "Interstellar"),
        ("Christopher Nolan", "DIRECTED", "The Dark Knight"),
        ("Christopher Nolan", "DIRECTED", "Oppenheimer"),
        ("Christopher Nolan", "DIRECTED", "Tenet"),
        
        ("Leonardo DiCaprio", "ACTED_IN", "Inception"),
        ("Matthew McConaughey", "ACTED_IN", "Interstellar"),
        ("Christian Bale", "ACTED_IN", "The Dark Knight"),
        ("Cillian Murphy", "ACTED_IN", "Inception"),
        ("Cillian Murphy", "ACTED_IN", "The Dark Knight"),
        ("Cillian Murphy", "ACTED_IN", "Oppenheimer"),
        
        ("Inception", "IS_GENRE", "Sci-Fi"),
        ("Inception", "IS_GENRE", "Thriller"),
        ("Interstellar", "IS_GENRE", "Sci-Fi"),
        ("The Dark Knight", "IS_GENRE", "Action"),
        ("Oppenheimer", "IS_GENRE", "Drama"),
        ("Tenet", "IS_GENRE", "Sci-Fi")
    ]

    for entity1, rel, entity2 in relationships:
        G.add_edge(entity1, entity2, relation=rel)
        
    return G

def get_graph_context(G, target_entity):
    """
    Extracts the immediate 'neighborhood' (connections) of a specific entity.
    """
    if target_entity not in G:
        return f"Entity '{target_entity}' not found in the knowledge graph."
        
    context = []
    # Find all direct connections to the target entity
    for neighbor in G.neighbors(target_entity):
        rel = G.edges[target_entity, neighbor]['relation']
        context.append(f"- {target_entity} {rel} {neighbor}")
        
        # Go one level deeper (Multi-hop)
        for second_neighbor in G.neighbors(neighbor):
            if second_neighbor != target_entity:
                rel2 = G.edges[neighbor, second_neighbor]['relation']
                context.append(f"  -> Because {neighbor} {rel2} {second_neighbor}")
                
    # Deduplicate and return as string
    return "\n".join(list(set(context)))

def query_graph(query: str, target_entity: str):
    G = build_knowledge_graph()
    
    print(f"\n🔍 Extracting Graph Sub-network for '{target_entity}'...")
    context = get_graph_context(G, target_entity)
    print(f"🕸️ Extracted Context:\n{context}\n")
    
    system_prompt = f"""
    You are a highly logical Streaming Service Recommendation Engine.
    You have been provided with a localized map of a Knowledge Graph. 
    Use ONLY the connections listed below to answer the user's query.
    Explain the logical connection (the path taken through the graph) in your answer.
    
    GRAPH CONNECTIONS:
    {context}
    """
    
    response = client.chat(
        model="gemma4:31b-cloud",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    )
    
    print(f"🤖 RECOMMENDATION ENGINE:\n{response['message']['content']}\n")

if __name__ == "__main__":
    # The Complex Query: We want to see if the AI can traverse nodes from an Actor -> Movie -> Director -> Another Movie
    user_query = "I really liked Cillian Murphy's performance in Oppenheimer. Can you recommend another Sci-Fi movie directed by the same person?"
    
    # We anchor the search on the main entity of interest
    query_graph(user_query, target_entity="Oppenheimer")