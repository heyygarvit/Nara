import sys
sys.path.append('src')
from advanced.DynamicGraph import DynamicGraph, Node



def test_node_operations():
    # Create two nodes
    a = Node(2)
    b = Node(3)

    # Test addition
    c = a + b
    assert c.value == 5, f"Expected 5, but got {c.value}"

    # Test multiplication
    d = a * b
    assert d.value == 6, f"Expected 6, but got {d.value}"

    # Test subtraction
    e = a - b
    assert e.value == -1, f"Expected -1, but got {e.value}"

    # Test division
    f = a / b
    assert f.value == 2/3, f"Expected 2/3, but got {f.value}"

    # Test in-place addition
    a += b
    assert a.value == 5, f"Expected 5, but got {a.value}"

    # Test in-place subtraction
    a -= b
    assert a.value == 2, f"Expected 2, but got {a.value}"

    # Test in-place multiplication
    a *= b
    assert a.value == 6, f"Expected 6, but got {a.value}"

    # Test in-place division
    a /= b
    assert a.value == 2, f"Expected 2, but got {a.value}"

    print("All node operations passed!")

def test_dynamic_graph():
    graph = DynamicGraph()

    # Add nodes to the graph
    a = graph.add_node(2)
    b = graph.add_node(3)

    # Perform operations
    c = a + b
    d = a * b

    # Backward pass
    graph.backward(c)

    # Zero gradients
    graph.zero_grad()
    for node in graph.nodes:
        assert node.grad == 0, f"Expected gradient 0, but got {node.grad}"

    print("All dynamic graph operations passed!")

if __name__ == "__main__":
    test_node_operations()
    test_dynamic_graph()
