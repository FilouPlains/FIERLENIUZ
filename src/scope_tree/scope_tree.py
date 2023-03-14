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

# [M]
from matplotlib import colormaps
# [T]
from tqdm import tqdm


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

    def plot_network(self, peitsch_code, min_size=10, max_size=50):
        max_size -= min_size

        log_size: np.ndarray = np.log(np.array(scope_tree.size) + 1)
        size: "list[float]" = list(np.array(log_size) / max(log_size)
                                   * max_size + min_size)

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
                          f"<br />Peitsch code {peitsch_code}</b>"),
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

        return net_plot


def get_domain(path: str, code: int) -> "list[str]":
    last_power = 0
    cluster = ""

    while code > 0:
        power = 0

        while 2 ** power < code:
            power += 1

        if power != 0:
            power -= 1

        if cluster != "":
            cluster += "0" * (last_power - power - 1)

        last_power = power
        code -= 2 ** power
        cluster += "1"

    domain_list: "list[str]" = []

    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING pyHCA FILE"):
            if line[0] != ">" and not line.startswith("cluster"):
                continue

            if line[0] == ">":
                domain: str = line.split()[0][1:]
                continue

            if cluster == line.strip().split()[-1]:
                domain_list += [domain]

    
    return domain_list


if __name__ == "__main__":
    PEITSCH_CODE: int = 5

    domain_list: "list[str]" = get_domain(
        "data/pyHCA_SCOPe_30identity_globular.out",
        PEITSCH_CODE
    )

    scope_tree: Scope = Scope("data/SCOPe_2.08_classification.txt")

    for domain in tqdm(list(domain_list), "   PARSING DOMAIN LIST"):
        scope_tree.add_domain(domain)

    scope_tree.plot_network(peitsch_code=PEITSCH_CODE).show()
