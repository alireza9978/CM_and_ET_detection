class Node:
    def __init__(self, data_value=None):
        self.data_value = data_value
        self.next_value = None


class SLinkedList:
    def __init__(self, head_node=None):
        self.head_value = head_node

    def at_beginning(self, new_data):
        new_node = Node(new_data)
        new_node.next_value = self.head_value
        self.head_value = new_node

    def in_betwen(self, middle_node, new_data):
        if middle_node is None:
            print("The mentioned node is absent")
            return
        new_node = Node(new_data)
        new_node.next_value = middle_node.nextval
        middle_node.next_value = new_node

    def at_end(self, new_data):
        new_node = Node(new_data)
        if self.head_value is None:
            self.head_value = new_node
            return
        temp_node = self.head_value
        while temp_node.next_value:
            temp_node = temp_node.nextval
        temp_node.next_value = new_node

    def add_all(self, new_list):
        temp_value = new_list.head_value
        while temp_value is not None:
            temp_temp_value = temp_value.next_value
            temp_value.next_value = self.head_value
            self.head_value = temp_value
            temp_value = temp_temp_value

    # Print the linked list
    def list_print(self):
        print_value = self.head_value
        while print_value is not None:
            print(print_value.data_value, end=", ")
            print_value = print_value.next_value

    def list(self):
        out = []
        temp_value = self.head_value
        while temp_value is not None:
            out.append(temp_value.data_value)
            temp_value = temp_value.next_value
        return out

    def size(self):
        count = 0
        temp_value = self.head_value
        while temp_value is not None:
            count += 1
            temp_value = temp_value.next_value
        return count
