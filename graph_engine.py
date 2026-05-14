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
    print("🕸️ Building Directed Knowledge Graph...")
    # FIX 1: Use a Directed Graph so relationships flow one way
    G = nx.DiGraph()

    movies = ["Inception", "Interstellar", "The Dark Knight", "Oppenheimer", "Tenet"]
    directors = ["Christopher Nolan"]
    actors = ["Leonardo DiCaprio", "Matthew McConaughey", "Christian Bale", "Cillian Murphy"]
    genres = ["Sci-Fi", "Action", "Drama", "Thriller"]

    for movie in movies: G.add_node(movie, type="Movie")
    for director in directors: G.add_node(director, type="Director")
    for actor in actors: G.add_node(actor, type="Actor")
    for genre in genres: G.add_node(genre, type="Genre")

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

def get_graph_context(G, target_entity, max_hops=3):
    """
    Extracts a sub-graph of all nodes within 'max_hops' of the target entity,
    preserving the strict direction of the relationships.
    """
    if target_entity not in G:
        return f"Entity '{target_entity}' not found in the knowledge graph."
        
    # FIX 2: N-Hop Extraction. 
    # We find all nodes within 3 hops (ignoring direction just for the search phase)
    undirected_G = G.to_undirected()
    relevant_nodes = list(nx.single_source_shortest_path_length(undirected_G, target_entity, cutoff=max_hops).keys())
    
    # We then create a mini-graph using only those nodes, which restores their strict directions
    subgraph = G.subgraph(relevant_nodes)
    
    context = []
    # Iterate through all edges in our new mini-graph
    for source, target, data in subgraph.edges(data=True):
        rel = data['relation']
        context.append(f"- {source} {rel} {target}")
                
    return "\n".join(sorted(list(set(context))))

def query_graph(query: str, target_entity: str):
    G = build_knowledge_graph()
    
    print(f"\n🔍 Extracting 3-Hop Graph Sub-network for '{target_entity}'...")
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
    user_query = "I really liked Cillian Murphy's performance in Oppenheimer. Can you recommend another Sci-Fi movie directed by the same person?"
    query_graph(user_query, target_entity="Oppenheimer")