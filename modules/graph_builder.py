import networkx as nx
from .relation_extractor import relation_extraction
from .common import get_nlp

def build_graph(text):
    graph = nx.DiGraph()

    # Add all named entities as nodes.
    doc = get_nlp()(text)
    for ent in doc.ents:
        graph.add_node(ent.text, label=ent.label_)

    relations = relation_extraction(text)
    for row in relations.to_dict("records"):
        graph.add_edge(
            row["Subject"],
            row["Object"],
            relation=row["Relation"],
        )

    return graph
