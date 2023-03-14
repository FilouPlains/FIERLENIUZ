"""This module contains an object to manipulate SCOPe classification.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "23s/02/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


# [I]
import igviz
# [N]
import networkx as net
import numpy as np
# [T]
import matplotlib.pyplot as plt

# [M]
from matplotlib import colormaps
# [T]
from tqdm import tqdm
# [S]
from sys import exit as sysexit


class Scope:
    """An object to manipulate SCOPe classification.
    """

    def __init__(
        self,
        path: str,
    ) -> None:
        """Instantiate a `Scope` object.

        Parameters
        ----------
        path : str
            A path to a SCOPe classification data file.
        """
        self.classification: "dict[str: str]" = {}
        self.tree: net.Graph = net.Graph()
        self.tree.add_node("0")

        cmap: object = colormaps["viridis"]
        self.viridis: object = cmap(np.linspace(0, 1, 5))
        self.color = [f"rgba{tuple(self.viridis[0])}"]

        self.size = [0]
        self.index = {"0": 0}

        # Parse a given SCOPe classification data file.
        with open(path, "r", encoding="utf-8") as file:
            for line in tqdm(list(file), "    PARSING SCOPe FILE"):
                # Skip the comment lines.
                if line[0] == "#":
                    continue

                split_line: "list[str]" = line.split()

                # Skip the line containing other things than domains.
                if split_line[3] == "-":
                    continue

                # Adding a domain to the classification dictionary.
                self.classification[split_line[3]] = split_line[2]

    def add_domain(self, domain: str) -> None:
        if domain not in self.classification:
            return None

        class_code: "str" = self.classification[domain].split(".")
        last: str = "0"

        self.size[0] += 1

        for i in range(len(class_code)):
            node: str = ".".join(class_code[0:i + 1])

            if node not in self.index:
                self.size += [0]
                self.index[node] = len(self.size) - 1
            else:
                self.size[self.index[node]] += 1

            if node not in self.tree:
                self.color += [f"rgba{tuple(self.viridis[i + 1])}"]
                self.tree.add_edge(last, node)

            last = node


if __name__ == "__main__":
    scope_tree: Scope = Scope("data/SCOPe_2.08_classification.txt")

    with open("/home/lrouaud/Téléchargements/147_code_scope.txt", "r",
              encoding="utf-8") as file:
        for line in tqdm(list(file), "   PARSING DOMAIN LIST"):
            scope_tree.add_domain(line.strip())

    log_size: np.ndarray = np.log(np.array(scope_tree.size) + 1)
    size: "list[float]" = list(np.array(log_size) / max(log_size) * 40 + 10)

    net_plot: object = igviz.plot(
        G=scope_tree.tree,
        title="",
        color_method=scope_tree.color,
        size_method=size,
        layout="kamada",
        colorscale=None,
        node_opacity=1,
        highlight_neighbours_on_hover=True,
        transparent_background=False,
        showlegend=True
    )

    net_plot.update_traces(
        marker_showscale=False,
        line_color="white",
        line_width=1
    )

    net_plot.update_layout(
        template="plotly_dark",
        legend_title=("<b>Network of SCOPe tree<br />classification for the"
                      "<br />Peitsch code 147</b>"),
        font=dict(size=14),
        margin=dict(l=30, r=30, t=30, b=30)
    )

    net_plot["data"][0]["name"] = "Edges"
    net_plot["data"][1]["name"] = "Nodes"

    shift_size: np.ndarray = (np.array(scope_tree.size) + 1).astype(str)

    hover_text: str = "<br />Leaf number: " \
        + ";<br />Leaf number: ".join(shift_size)

    hover: np.ndarray = np.array(hover_text.split(";"))

    new_hover: np.ndarray = np.char.add(
        np.array(net_plot["data"][1]["hovertext"]).astype(str),
        hover.astype(str)
    )

    net_plot["data"][1]["hovertext"] = new_hover

    net_plot.show()
