class Node:
    def __init__(self, value):
        self.value = value
        self.prev = None
        self.next = None
    
    def __repr__(self):
        return f"Node({self.value})"
    

def Build_linked_list(s: str, byte_shuffle: list[int] = None) -> list[Node]:
    DLL = []
    ids = list(s.encode('utf-8'))
    for id in ids:
        if byte_shuffle is not None and id < 256:
            id = byte_shuffle[id]
        new_node = Node(id)
        DLL.append(new_node)
    for i in range(len(DLL) - 1):
        DLL[i].next = DLL[i + 1]
        DLL[i + 1].prev = DLL[i]
    return DLL