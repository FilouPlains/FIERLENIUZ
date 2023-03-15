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
# [P]
import plotly.graph_objects as go
# [S]
from statistics import median
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
        # To go from domain to SCOPe classification code.
        self.classification: "dict[str: str]" = {}

        # Create the network.
        self.tree: net.Graph = net.Graph()
        self.tree.add_node("0")

        # To use viridis colour.
        cmap: object = colormaps["viridis"]
        self.viridis: object = cmap(np.linspace(0, 1, 5))
        self.color = [f"rgba{tuple(self.viridis[0])}"]

        # To obtain the number of leaves in a given nodes.
        self.size = [-1]
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
        """Add a domain to the network graph.

        Parameters
        ----------
        domain : str
            The domain to add.
        """
        # To check if the domain is in the classification dictionary.
        if domain not in self.classification:
            return None

        # Transform `a.1.1.1` in a list like `["a", "1", "1", "1"]`
        class_code: "str" = self.classification[domain].split(".")
        last: str = "0"
        self.size[0] += 1

        # To all SCOPe level of a given class like : a; a.1; a.1.1 and a.1.1.1
        for i in range(len(class_code)):
            # Check a level.
            node: str = ".".join(class_code[0:i + 1])

            # Is this a new node to add ?
            if node not in self.index:
                self.size += [0]
                self.index[node] = len(self.size) - 1
            else:
                # Increase node's size.
                self.size[self.index[node]] += 1

            # If it's a new node, add its colour to the colour list.
            if node not in self.tree:
                self.color += [f"rgba{tuple(self.viridis[i + 1])}"]
                self.tree.add_edge(last, node)

            last = node

    def plot_network(
        self,
        peitsch_code: "int | str",
        min_size: int = 10,
        max_size: int = 50
    ) -> object:
        """Plot a network graph using plotly.

        Parameters
        ----------
        peitsch_code : int | str
            The peitsch code used to do the network.
        min_size : int, optional
            The minimal size of a plotted node, by default 10.
        max_size : int, optional
            The maximal size of a plotted node, by default 50.

        Returns
        -------
        object
            A plotly plot.
        """
        max_size -= min_size

        # Log the value to have a better view on the nodes' size diversity.
        log_size: np.ndarray = np.log(np.array(self.size) + 1)
        size: "list[float]" = list(np.array(log_size) / max(log_size)
                                   * max_size + min_size)

        # Craft the network.
        net_plot: object = igviz.plot(
            G=self.tree,
            title="",
            color_method=self.color,
            size_method=size,
            layout="kamada",
            colorscale=None,
            node_opacity=1,
            highlight_neighbours_on_hover=True,
            transparent_background=False,
            showlegend=True
        )

        # Modify traces attributs.
        net_plot.update_traces(
            marker_showscale=False,
            line_color="white",
            line_width=1
        )

        # Modify the general layout.
        net_plot.update_layout(
            template="plotly_dark",
            legend_title=("<b>Network of SCOPe tree<br />classification for the"
                          f"<br />Peitsch code {peitsch_code}</b>"),
            font=dict(size=14),
            margin=dict(l=30, r=30, t=30, b=30)
        )

        # Modify legends' labels.
        net_plot["data"][0]["name"] = "Edges"
        net_plot["data"][1]["name"] = "Nodes"

        # Add the size in the hover text.
        shift_size: np.ndarray = (np.array(self.size) + 1).astype(str)

        hover_text: str = "<br />Leaf number: " \
            + ";<br />Leaf number: ".join(shift_size)

        hover: np.ndarray = np.array(hover_text.split(";"))

        new_hover: np.ndarray = np.char.add(
            np.array(net_plot["data"][1]["hovertext"]).astype(str),
            hover.astype(str)
        )

        net_plot["data"][1]["hovertext"] = new_hover

        # To set x and y lim of the circle legends.
        x_pos: float = max(list(net_plot["data"][1]["x"])) * 1.25
        y_min: float = median(list(net_plot["data"][1]["y"]))
        y_max: float = max(list(net_plot["data"][1]["y"]))
        y_pos: np.ndarray = np.linspace(y_min, y_max, 5)

        # Log the value to have a better view on the nodes' size diversity.
        circle_size: np.ndarray = np.linspace(min(scope_tree.size),
                                              max(scope_tree.size), 5)
        log_size: np.ndarray = np.log(circle_size + 1)
        size: "list[float]" = np.array(log_size) / max(log_size) * 40 + 10

        # Add the circle legend (node size explanation).
        side_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[x_pos] * 5,
            y=y_pos,
            marker=dict(color="white", size=size, opacity=1),
            hoverinfo="skip",
            name="Number of leaf"
        )

        net_plot.add_trace(side_plot)

        # Add annotation of the circle to the plot.
        for i, y in enumerate(y_pos):
            net_plot.add_annotation(
                xanchor="center",
                yanchor="middle",
                x=x_pos * 1.05,
                y=y,
                text=f"<b>{circle_size[i] + 1:.0f}</b>",
                showarrow=False,
                font_color="white",
                align="center"
            )

        return net_plot


def get_domain(path: str, code: int) -> "list[str]":
    """Get a domain list from a given Peitsch code.

    Parameters
    ----------
    path : str
        Path to pyHCA output file.
    code : int
        The Peitsch code.

    Returns
    -------
    list[str]
        The list of extracted domains.
    """
    last_power = 0
    cluster = ""

    # To translate a Peitsch code into a hydrophobic cluster.
    while code > 0:
        power = 0

        # When 2 ** power is greater than the code, power - 1 is the index where
        # we can put a 1 into the hydrophobic cluster list/text.
        while 2 ** power < code:
            power += 1

        # When we reach the last position, we don't have to do `-1` to the
        # power.
        if power != 0:
            power -= 1

        # Done only when the first `1` is add to the cluster.
        if cluster != "":
            cluster += "0" * (last_power - power - 1)

        last_power = power
        code -= 2 ** power
        cluster += "1"

    domain_list: "list[str]" = []

    # Parse the file to check if a given cluster is in a domain.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING pyHCA FILE"):
            # Only checking domain ID and cluster line.
            if line[0] != ">" and not line.startswith("cluster"):
                continue

            # Domaine line.
            if line[0] == ">":
                domain: str = line.split()[0][1:]
                continue

            # Cluster line: if we found the exact same cluster, add it to our
            # list.
            if cluster == line.strip().split()[-1]:
                domain_list += [domain]

    # Return a list of unique elements, as far as a same Peitsch code can be
    # multiple time in the same domain.
    return list(set(domain_list))


if __name__ == "__main__":
    PEITSCH_CODE: int = 147

    domain_list: "list[str]" = get_domain(
        "data/pyHCA_SCOPe_30identity_globular.out",
        PEITSCH_CODE
    )

    scope_tree: Scope = Scope("data/SCOPe_2.08_classification.txt")

    for i, domain in tqdm(enumerate(domain_list), "   PARSING DOMAIN LIST"):
        scope_tree.add_domain(domain)

    plot_distribution: go.FigureWidget = scope_tree.plot_network(
        peitsch_code=PEITSCH_CODE
    ).show()
