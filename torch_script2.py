import torch
import torchvision
from torch.profiler import profile, ProfilerActivity
import graphviz
import networkx as nx
# import torch._C.Node


def common_member(a: list, b: list) -> bool:
    """
    check if list a and list b have common member
    """
    a_set = set(a)
    b_set = set(b)
    if len(a_set.intersection(b_set)) > 0:
        return True
    return False


class TorchScriptNode:
    def __init__(self, node):
        self.node = node

        # node name
        self.node_str = str(self.node)
        self.unique = self.node_str.split(' ')[0][1:].replace('.', '')  # unique IR identifier
        self.ofmap_shape = self._find_ofmap_shape()
        self.node_op = self.node.kind()  # node operation type

        # the input and output nodes connected to this node
        self.outputs = [o.unique() for o in node.outputs()]
        self.inputs = [i.unique() for i in node.inputs()]

    def __str__(self):
        return self.unique + "\t" + str(self.ofmap_shape) + " "

    def _find_ofmap_shape(self) -> list:
        ind_left = self.node_str.find("Float(")
        ind_right = self.node_str.find("strides=[")
        str_with_shape = self.node_str[ind_left + 6: ind_right - 2]
        ofmap_shape = [int(num) for num in str_with_shape.split(",")]
        return ofmap_shape


class TorchScriptGraph:
    def __init__(self, torch_script_model):
        # data type
        self.aten_node_list = []  # data type: TorchScriptNode
        self.torch_script_model = torch_script_model
        self.graph = graphviz.Digraph()

        # run some script
        self._construct_graph()

        # iterator pointer
        self.__iter_ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        return aten_node_list one by one
        """
        if self.__iter_ptr < len(self.aten_node_list):
            ret = self.aten_node_list[self.__iter_ptr]
            self.__iter_ptr += + 1
            return ret
        else:
            __iter_ptr = 0
            raise StopIteration

    def _construct_graph(self):
        # add node to graph
        for node in self.torch_script_model.inlined_graph.nodes():
            if "aten" in node.kind():
                # construct torch script node
                ts_node = TorchScriptNode(node)
                # append to aten_node_list and graphviz graph
                self.aten_node_list.append(ts_node)
                self.graph.node(ts_node.unique, ts_node.node_op + f"({ts_node.ofmap_shape})")

        # add edge to graph
        # this is done by looping through nodes to check if two nodes have
        # "node1.outputs" intersects with "nodes2.inputs"
        try:
            for i, ts_node in enumerate(self.aten_node_list):
                for j in range(i + 1, len(self.aten_node_list)):
                    ts_node1 = ts_node
                    ts_node2 = self.aten_node_list[j]
                    if common_member(ts_node1.outputs, ts_node2.inputs):
                        self.graph.edge(ts_node1.unique, ts_node2.unique)
        except IndexError:
            print(f"The length of torch_script_node_list in TorchScriptGraph is not >= 2, "
                  f"it is {len(self.aten_node_list)}")

    def annotate_prof_result(self, prof_node_list: list):
        pass


if __name__ == '__main__':
    # construct model
    model_name = "rn18"
    model = torchvision.models.resnet18().cuda()
    example = torch.rand(1, 3, 224, 224).cuda()
    torch_script_model = torch.jit.trace(model, example)

    """
    Profiling ------------------------------------------------------------------------------------------
    """
    # profiling
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        torch_script_model.forward(example)

    # extract root node and its attributes
    prof_node_list = []  # each {"prof_id", "op", "ifmap_shape", "cuda_time_total"}
    for node in prof.profiler.function_events:
        if node.cpu_parent is not None and node.cpu_parent.name == "forward":
            node_attr = {
                "prof_id": str(node.id),
                "op": node.name,
                "ifmap_shape": node.input_shapes[0],
                "cuda_time_total": node.node_cuda_time_total
            }
            prof_node_list.append(node_attr)

    """
    Construct Graph Dynamically ------------------------------------------------------------------------
    """
    # construct graph
    torch_script_graph = TorchScriptGraph(torch_script_model)
    torch_script_graph.annotate_prof_result(prof_node_list)

    for node in torch_script_graph:
        print(node)

    # render
    torch_script_graph.graph.render(model_name + '.dot', view=True)
    torch_script_model.save(model_name + ".pt")
    pass
