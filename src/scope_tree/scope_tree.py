"""This module contains an object to manipulate SCOPe classification.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "04/04/2023"
__version__ = "2.0.0"
__copyright__ = "CC BY-SA"


# ==================================================

# To skip all warnings link to numpy module version.

# [W]
from warnings import simplefilter
simplefilter("ignore", DeprecationWarning)

# ==================================================


# [ FULL IMPORT ]
# [I]
import igviz
# [N]
import networkx as net
import numpy as np


# [ PARTIAL IMPORT ]
# [A]
from arg_parser import parsing
# [D]
from datetime import datetime
# [M]
from matplotlib import colormaps
# [O]
from os.path import join
# [P]
import plotly.graph_objects as go
# [S]
from statistics import median
from sys import path
# [T]
from tqdm import tqdm


# [ EXTERNAL IMPORT ]
# Checking parent directory.
path.append("src/embeddings/")

# [C]
from context_analyzer import PairewiseContextAnalyzer, center_context


class ScopeTree:
    """An object to manipulate SCOPe classification.
    """

    def __init__(
        self,
        path: str,
        order_matrix: np.ndarray,
        unorder_matrix: np.ndarray,
        index_dict: "dict[str : int]"
    ) -> None:
        """Instantiate a `Scope` object.

        Parameters
        ----------
        path : `str`
            A path to a SCOPe classification data file.
        order_matrix : `np.ndarray`
            Matrix containing the computed context pairwise using O(NP)
            algorithm.
        unorder_matrix : `np.ndarray`
            Matrix containing the computed context pairwise using Bray-Curtis
            distance.
        index_dict : `dict[str : int]`
            Index liking domain (as keys) to index in the matrix (as int).
        """
        # To go from domain to SCOPe classification code.
        self.classification: "dict[str: str]" = {}

        # Create the network. A node = A SCOPe class code. A edge = Link between
        # to class.
        self.tree: net.Graph = net.Graph()
        # Initialize the network with the node "0".
        self.tree.add_node("0")

        # To use viridis colour.
        cmap: object = colormaps["viridis"]
        self.viridis: object = cmap(np.linspace(0, 1, 5))
        self.color = [f"rgba{tuple(self.viridis[0])}"]

        # To obtain the number of leaves in a given nodes.
        self.size: "list[int]" = [-1]
        # To link SCOPe code classification (as key) to their index in other
        # list.
        self.index: "dict[str : int]" = {"0": 0}
        # To obtain the number of total leaves in each node for the SCOPe
        # classificiation.
        self.absolute_size: "dict[str : int]" = {"0": 0}
        # To order the absolute size in function of our node.
        self.order_abs_size: "list[int]" = []
        # Labels to display on the network plot.
        self.label: "list[str]" = []

        # Set up attributes with given parameters.
        self.order_matrix: np.ndarray = order_matrix
        self.unorder_matrix: np.ndarray = unorder_matrix
        self.index_dict: "dict[str : int]" = index_dict

        # To get the index list ordered in function of which domain we compare.
        self.matrix_index: "list[list[int]]" = [[]]

        # Parse a given SCOPe classification data file.
        with open(path, "r", encoding="utf-8") as file:
            for line in tqdm(list(file), "    PARSING SCOPe FILE"):
                # Skip the comment lines.
                if line[0] == "#":
                    continue

                split_line: "list[str]" = line.split()

                # Set up a new size.
                if len(split_line[2]) <= 5:
                    if split_line[2] not in self.absolute_size:
                        self.absolute_size[split_line[2]] = 0

                # Skip the line containing other things than domains.
                if split_line[3] == "-":
                    continue

                # Every time, adding one to the original node.
                self.absolute_size["0"] += 1
                split_class: "list[str]" = split_line[2].split(".")

                # Parse all sub SCOPe class. So for `a.1.1.1`, we parse `a`,
                # `a.1.1` and `a.1.1.1`.
                for i in range(4):
                    normal_class: str = ".".join(split_class[:i + 1])

                    # Set up a new size.
                    if normal_class not in self.absolute_size:
                        self.absolute_size[normal_class] = 1
                    else:
                        self.absolute_size[normal_class] += 1

                # Adding a domain to the classification dictionary.
                self.classification[split_line[3]] = split_line[2]

        # Adding 0 for the ordered absolute_size.
        self.order_abs_size += [self.absolute_size["0"]]

    def __iadd__(self, domain: str) -> object:
        """Add a domain to the network graph.

        Parameters
        ----------
        domain : `str`
            The domain to add.

        Returns
        -------
        object
            self.
        """
        # To check if the domain is in the classification dictionary.
        if domain not in self.classification or domain not in self.index_dict:
            print(f"[Warn##] Domain [{domain}] is not in the SCOPe "
                  "classification dictionary, skipping...")
            return self

        # Transform `a.1.1.1` in a list like `["a", "1", "1", "1"]`
        class_code: "str" = self.classification[domain].split(".")
        # Adding parameters to the original nodes.
        last: str = "0"
        self.size[0] += 1
        self.matrix_index[0] += list(self.index_dict[domain])
        self.label += [""]

        # To all SCOPe level of a given class like : a; a.1; a.1.1 and a.1.1.1
        for i in range(len(class_code)):
            # Check a level.
            node: str = ".".join(class_code[0:i + 1])

            # Is this a new node to add ?
            if node not in self.index:

                # Adding the size of a given node.
                self.order_abs_size += [self.absolute_size[node]]

                # So, when we plot label, we only display first SCOPe class
                # level (a, b...).
                if len(node) <= 1:
                    self.label += [node]
                else:
                    self.label += [""]

                # Increasing the total size for a given node.
                self.size += [1]
                # Setting index to extract context values from matrix.
                self.matrix_index += [list(self.index_dict[domain])]
                # Adding a index to set data of previous set-up list.
                self.index[node] = len(self.size) - 1
            else:
                # Increase node's size.
                self.size[self.index[node]] += 1
                # Adding index to extract context values from matrix.
                self.matrix_index[self.index[node]] += list(
                    self.index_dict[domain]
                )

            # If it's a new node, add its colour to the colour list.
            if node not in self.tree:
                self.color += [f"rgba{tuple(self.viridis[i + 1])}"]
                self.tree.add_edge(last, node)

            # So each time, we change the edge pairing like so:
            # - `[0, a]`
            # - `[a, a.1]`
            # - `[a.1, a.1.1]`
            # - `[a.1.1, a.1.1.1]`
            last = node

        return self

    def plot_network(
        self,
        peitsch_code: "int | str",
        min_size: int = 10,
        max_size: int = 50,
        to_percent: bool = True
    ) -> object:
        """Plot a network graph using plotly.

        Parameters
        ----------
        peitsch_code : `int | str`
            The peitsch code used to do the network.
        min_size : `int`, optional
            The minimal size of a plotted node, by default 10.
        max_size : `int`, optional
            The maximal size of a plotted node, by default 50.
        to_percent : `bool`, optional
            If the node size is linked to the total number of leaves or if we
            compute a percentage:

        ```
                          number of leaves in a given nodes
        percentage = 1 - ——————————————————————————————————— x 100
                            total leaves in a given nodes
        ```

        Returns
        -------
        `object`
            A plotly plot.
        """
        # To set list of colors.
        unorder_color: "list[int]" = []
        order_color: "list[int]" = []

        # Parsing (yet again) all nodes. More precisely, matrix index for a
        # given node.
        for i, m_i in enumerate(tqdm(self.matrix_index,
                                     "  COMPUTING COLORATION")):
            # Create a list of unique elements.
            m_i = list(set(m_i))

            # No comparison to do (only one domaine in the nodes). We set up the
            # future colours to -1. (And yes, I used both color and colour).
            if len(m_i) == 1:
                order_color += [-1]
                unorder_color += [-1]
            else:
                # Get all context value in the matrix and sum them.
                mat_sum: np.ndarray = np.sum(self.order_matrix[m_i, :][:, m_i])
                # Get the size of the sub-matrix.
                length: int = self.order_matrix[m_i, :][:, m_i].shape[0]
                # To mean value, we take the whole matrix size less the
                # diagonal size which is, here, `length`.
                to_mean: int = length ** 2 - length

                # Mean for ordered context.
                order_color += [mat_sum / to_mean]

                # Get all context value in the matrix and sum them.
                mat_sum: np.ndarray = np.sum(
                    self.unorder_matrix[m_i, :][:, m_i]
                )
                # Get the size of the sub-matrix.
                length: int = self.order_matrix[m_i, :][:, m_i].shape[0]
                # To mean value, we take the whole matrix size less the
                # diagonal size which is, here, `length`.
                to_mean: int = length ** 2 - length

                # Mean for unordered context.
                unorder_color += [mat_sum / to_mean]

        # Change all `-1` to gray AND put to these one no border. Do it for
        # both order and unorder computed values.
        order_color = np.array(order_color)

        # Define the colour palette.
        cmap: object = colormaps["inferno_r"]
        v_order: np.ndarray = cmap(order_color)
        # Transform [0, 1] colour values to [0, 255].
        v_order *= np.array([255, 255, 255, 1])
        # Transform values to rgba format.
        v_order = np.array(list(map(lambda line: f"rgba{tuple(line)}",
                                    v_order)))
        # Change `-1` values to gray.
        v_order[order_color == -1] = "rgba(150, 150, 150, 0.5)"
        # Save the values as a list.
        v_order = list(v_order)

        unorder_color = np.array(unorder_color)

        # Define the colour palette.
        cmap: object = colormaps["inferno_r"]
        v_unorder: np.ndarray = cmap(unorder_color)
        # Transform [0, 1] colour values to [0, 255].
        v_unorder *= np.array([255, 255, 255, 1])
        # Transform values to rgba format.
        v_unorder = np.array(list(map(lambda line: f"rgba{tuple(line)}",
                                      v_unorder)))
        # Change `-1` values to gray.
        v_unorder[unorder_color == -1] = "rgba(150, 150, 150, 0.5)"
        # Save the values as a list.
        v_unorder = list(v_unorder)

        # Set up gray border to all nodes (so we see them better).
        border_color: np.ndarray = np.array(
            ["rgba(68, 68, 68, 1)"] * len(v_unorder),
            dtype=str
        )
        # Change nodes' border with a context value of `-1` with no fully
        # transparent (in other words, `None`).
        border_color[unorder_color == -1] = "rgba(0, 0, 0, 0)"
        border_color = list(border_color)

        max_size -= min_size

        # Set up the size to percentage.
        if to_percent:
            # Percentage of the leaves size values.
            size: "list[float]" = list(np.array(self.size) /
                                       np.array(self.order_abs_size) * max_size
                                       + min_size)
        else:
            # Log the value to have a better view on the nodes' size diversity.
            log_size: np.ndarray = np.log(np.array(self.size) + 1)
            size: "list[float]" = list(np.array(log_size) / max(log_size)
                                       * max_size + min_size)

        # Craft the network.
        net_plot: object = igviz.plot(
            G=self.tree,
            title="",
            color_method=v_unorder,
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
            line_color="#DDD",
            line_width=1,
            marker_line=dict(color=border_color, width=1)
        )

        # Modify the general layout.
        net_plot.update_layout(
            plot_bgcolor="white",
            legend_title=("<b>Network of SCOPe tree<br />classification for the"
                          f"<br />Peitsch code {peitsch_code}</b>"),
            font=dict(size=14),
            margin=dict(l=30, r=30, t=30, b=30),
            coloraxis_showscale=True,
            # Add the menu to change between order and unorder context values.
            updatemenus=[dict(
                buttons=[
                    dict(
                        method="restyle",
                        label="Bray-Curtis",
                        args=[
                            "marker",
                            dict(color=v_unorder, size=size, opacity=1,
                                 line=dict(color=border_color, width=1)),
                            [1]
                        ]
                    ),
                    dict(
                        method="restyle",
                        label="O(NP)",
                        args=[
                            "marker",
                            dict(color=v_order, size=size, opacity=1,
                                 line=dict(color=border_color, width=1)),
                            [1]
                        ]
                    )
                ],
                type="dropdown",
                direction="up",
                showactive=True,
                x=1.01,
                xanchor="left",
                y=0,
                yanchor="bottom",
                font_color="black"
            )]
        )

        # Modify legends' labels.
        net_plot["data"][0]["name"] = "Edges"
        net_plot["data"][1]["name"] = "Nodes"

        # Add data in the hover text.
        order_hover: np.ndarray = np.array(order_color) * 100
        unorder_hover: np.ndarray = np.array(unorder_color) * 100
        percent_hover: np.ndarray = (np.array(size) - 10) / 4 * 10

        new_hover: "list[str]" = []

        for i, shift in enumerate(tqdm(self.size, "       MODIFYING HOVER")):
            # To take in consideration if a order context have been computed.
            if order_hover[i] < 0:
                order: str = f"Order context: None<br>"
            else:
                order: str = f"Order context: {order_hover[i]:.1f}<br>"

            # To take in consideration if a unorder context have been computed.
            if unorder_hover[i] < 0:
                unorder: str = f"Unorder context: None<br>"
            else:
                unorder: str = f"Unorder context: {unorder_hover[i]:.1f}<br>"

            # Add all text data.
            new_hover += [f"Leaf number: {shift}<br>"
                          f"{unorder}{order}"
                          f"Relative size: {percent_hover[i]:.1f} %<br>"
                          # Append old hovertext at the end.
                          f"{net_plot['data'][1]['hovertext'][i]}"]

        # Change hover information.
        net_plot["data"][1]["hovertext"] = new_hover

        # To set x and y lim of the circle legends.
        x_pos: float = max(list(net_plot["data"][1]["x"])) * 1.25
        y_min: float = median(list(net_plot["data"][1]["y"])) * 1.2
        y_max: float = max(list(net_plot["data"][1]["y"]))
        y_pos: np.ndarray = np.linspace(y_min, y_max, 5)

        # Log the value to have a better view on the nodes' size diversity.
        circle_size: np.ndarray = np.linspace(min(size), max(size), 5)

        # Add the circle legend (node size explanation).
        side_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[x_pos] * 5,
            y=y_pos,
            marker=dict(color="#000", size=circle_size, opacity=1),
            hoverinfo="skip",
            name="Relative size to SCOPe<br />count"
        )

        # Add the circle legend (node size explanation).
        legend_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[None],
            y=[None],
            marker=dict(color="rgba(150, 150, 150, 0.5)", size=50, opacity=1),
            hoverinfo="skip",
            name="Context not computed"
        )

        # To set up ticks in the colorbar.
        ticks: "list[int]" = list(np.linspace(
            0,
            100,
            5
        ).astype(int))

        # Add a dummy plot to add a colorbar.
        colorbar_plot: go.Scatter = go.Scatter(
            mode="markers",
            x=[None],
            y=[None],
            marker=dict(
                color="rgba(0, 0, 0, 0)",
                opacity=0,
                colorbar=dict(
                    title="Distance (%)",
                    title_font=dict(size=14),
                    tickvals=ticks,
                    ticktext=ticks,
                    lenmode="pixels",
                    len=150,
                    yanchor="bottom",
                    y=0.09,
                ),
                colorscale="inferno_r",
                cmin=0,
                cmax=100
            ),
            name=""
        )

        # Append all dummy plot for the legend to the main plot.
        net_plot.add_trace(side_plot)
        net_plot.add_trace(legend_plot)
        net_plot.add_trace(colorbar_plot)

        # To add the legend of the grey transparent nodes.
        for i, y in enumerate(y_pos):
            net_plot.add_annotation(
                xanchor="center",
                yanchor="middle",
                x=x_pos * 0.85,
                y=y,
                text=f"<b>{(circle_size[i] - 10) / 4 * 10:.0f}</b>",
                showarrow=False,
                font_color="#000",
                align="center"
            )

        # Add a black border to the plot.
        net_plot.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0,
            y0=0,
            x1=1.0,
            y1=1.0,
            opacity=1,
            line=dict(color="rgba(0, 0, 0, 1)", width=1)
        )

        # Add label like `a`, `b`... Hover their respective nodes.
        for i, label in enumerate(net_plot["data"][1]["hovertext"]):
            node_name: str = label.split("<br>")[-2]

            if node_name[-1] == "0":
                continue
            if len(node_name) != 7:
                continue
            else:
                net_plot.add_annotation(
                    xanchor="center",
                    yanchor="middle",
                    x=net_plot["data"][1]["x"][i],
                    y=net_plot["data"][1]["y"][i],
                    text=f"<b> {node_name[-1]} </b>",
                    showarrow=False,
                    font_color="black",
                    align="center",
                    font=dict(size=14),
                    bgcolor="rgba(255, 255, 255, 0.6)"
                )

        return net_plot


def get_domain(path: str, code: int) -> "tuple":
    """Get a domain list from a given Peitsch code.

    Parameters
    ----------
    path : `str`
        Path to pyHCA output file.
    code : `int`
        The Peitsch code.

    Returns
    -------
    `list[str]`
        The list of extracted domains.
    """
    last_power = 0
    cluster = ""

    # To translate a Peitsch code into a hydrophobic cluster.
    while code > 0:
        power = 0

        # When 2 ** power is greater than the code, power - 1 is the index
        # where we can put a 1 into the hydrophobic cluster list/text.
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
    context_dict: "dict[str : list]" = {}
    in_domain: bool = False
    domain: str = ""

    # Parse the file to check if a given cluster is in a domain.
    with open(path, "r", encoding="utf-8") as file:
        for line in tqdm(list(file), "    PARSING pyHCA FILE"):
            # Only checking domain ID and cluster line.
            if line[0] != ">" and not line.startswith("cluster"):
                continue

            # Domaine line.
            if line[0] == ">":
                if in_domain:
                    # Add the centered context to the dictionary.
                    context_dict[domain] = list(center_context(
                        context=np.array([context]),
                        window=10,
                        center=cluster,
                        add_gap=False,
                        keep_center=False
                    ))

                domain = "d" + line.split()[0][2:]
                context: "list[str]" = []
                in_domain = False
                continue

            hc: str = line.strip().split()[-1]

            # Skip to small clusters.
            if len(hc) <= 2:
                continue
            context += [hc]

            # Cluster line: if we found the exact same cluster, add it to our
            # list.
            if cluster != hc:
                continue

            in_domain |= True
            domain_list += [domain]

    domain_list = list(set(domain_list))
    domain_list.sort()

    length: int = 0

    # To get the full length of the number of context per domain, because we
    # have a list of list.
    for content in context_dict.values():
        length += len(content)

    keys: "list[str]" = list(context_dict.keys())

    # Set up the futur context matrix.
    order_matrix: np.ndarray = np.zeros((length, length), dtype=float)
    unorder_matrix: np.ndarray = np.zeros((length, length), dtype=float)

    index_dict: "dict[str, int]" = {}
    domain_dict: "dict[str, int]" = {}

    index: int = 0

    # Loop in the half matrix.
    for shift_i, i in enumerate(tqdm(keys[:-1], "    MATRIX COMPUTATION")):
        for j in keys[shift_i + 1:]:
            order: "list[float]" = []
            unorder: "list[float]" = []

            # The whole context (domain 1 vs domain 2).
            context: "list[str]" = context_dict[i] + context_dict[j]
            # Domains names linked to `context` (number and order).
            d_cont: "list[str]" = [i] * len(context_dict[i]) \
                + [j] * len(context_dict[j])
            # ID for the domain linked to `context` (number and order).
            n_cont: "list[int]" = list(range(len(context_dict[i]))) \
                + list(range(len(context_dict[j])))

            # Loop in the half matrix of the context in the domain. We do like
            # so as far as we can have multiple context in a domain.
            for pos_a, a in enumerate(context[:-1]):
                if len(a) <= 0:
                    continue

                # Generate a unique key which take in consideration the domain
                # and the multiple context per domain.
                dict_key_a: str = f"{d_cont[pos_a]}_{n_cont[pos_a]}"

                # Add the key to the dictionary.
                if dict_key_a not in index_dict:
                    index_dict[dict_key_a] = index
                    index += 1

                i_a: int = index_dict[dict_key_a]

                # Add matrix index.
                if d_cont[pos_a] not in domain_dict:
                    domain_dict[d_cont[pos_a]] = [i_a]
                else:
                    domain_dict[d_cont[pos_a]] += [i_a]

                for pos_b, b in enumerate(context[pos_a + 1:]):
                    if len(b) <= 0:
                        continue

                    pos_b += 1 + pos_a

                    # Generate a unique key which take in consideration the
                    # domain and the multiple context per domain.
                    dict_key_b: str = f"{d_cont[pos_b]}_{n_cont[pos_b]}"

                    # Add the key to the dictionary.
                    if dict_key_b not in index_dict:
                        index_dict[dict_key_b] = index
                        index += 1

                    i_b: int = index_dict[dict_key_b]

                    # Add matrix index.
                    if d_cont[pos_b] not in domain_dict:
                        domain_dict[d_cont[pos_b]] = [i_b]
                    else:
                        domain_dict[d_cont[pos_b]] += [i_b]

                    # Compute order and unorder context.
                    Context: object = PairewiseContextAnalyzer(a, b)
                    Context.compute_distance()

                    order = Context.distance[0]
                    unorder = Context.distance[1]

                    order_matrix[i_a][i_b] = order
                    order_matrix[i_b][i_a] = order
                    unorder_matrix[i_a][i_b] = unorder
                    unorder_matrix[i_b][i_a] = unorder

    # Return a list of unique elements, as far as a same Peitsch code can be
    # multiple time in the same domain.
    return domain_list, [order_matrix, unorder_matrix, domain_dict]


if __name__ == "__main__":
    introduction: str = """
    ███████╗██╗███████╗██████╗ ██╗     ███████╗███╗   ██╗██╗██╗   ██╗███████╗
    ██╔════╝██║██╔════╝██╔══██╗██║     ██╔════╝████╗  ██║██║██║   ██║██╔════╝
    █████╗  ██║█████╗  ██████╔╝██║     █████╗  ██╔██╗ ██║██║██║   ██║███████╗
    ██╔══╝  ██║██╔══╝  ██╔══██╗██║     ██╔══╝  ██║╚██╗██║██║██║   ██║╚════██║
    ██║     ██║███████╗██║  ██║███████╗███████╗██║ ╚████║██║╚██████╔╝███████║
    ╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚══════╝
    """

    print(introduction)
    argument: "dict[str: str|int]" = parsing(is_directory=True)

    if not argument["integer"]:
        peitsch: np.ndarray = np.transpose(np.load(
            argument["input"],
            allow_pickle=True
        ))[0].astype("int64")
    else:
        peitsch: list = [argument["input"]]

    date: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # To initialize the plot distribution (box + violin).
    plot_distribution: object = go.Figure()

    cmap = colormaps["viridis"]

    # Fill the plot with transparent colours.
    color: object = cmap(np.linspace(0, 1, len(peitsch)))
    fill: object = np.array(color)
    fill[:, -1] = 0.35

    # Parse all Peitsch code.
    for i, code in enumerate(peitsch):
        # Get all data and context values in matrix format.
        domain_list, data_list = get_domain(argument["context"], code)

        # Instantiate a Scope object
        scope_tree: ScopeTree = ScopeTree(
            argument["scope"],
            order_matrix=data_list[0],
            unorder_matrix=data_list[1],
            index_dict=data_list[2]
        )

        # Add domains to the tree.
        for domain in tqdm(domain_list, "   PARSING DOMAIN LIST"):
            scope_tree += domain

        # Parse all SCOPe class to output distributions in function of them.
        # "0" refers to the tree's root.
        for j, key in enumerate(["0", "a", "b", "c", "d"]):
            # Change the label and legend showing if we are at the root or not.
            if key == "0":
                name_label: str = f"<b>{code} (all)"
                show_legend: bool = True
            else:
                name_label: str = f"{code} ({key})"
                show_legend: bool = False

            # Distribution for the order context.
            m_i: set = list(set(scope_tree.matrix_index[scope_tree.index[key]]))
            matrix: np.ndarray = data_list[0][m_i, :][:, m_i]
            x: np.ndarray = matrix[np.triu_indices(matrix.shape[0],
                                   k=1)] * 100

            if x.shape[0] == 0:
                continue

            # Those next plot are hide by default.
            # Add a violin plot.
            plot_distribution.add_trace(go.Violin(
                y=x,
                x0=f"{name_label} [O(NP)]</b>",
                name=code,
                legendgroup="ONP",
                legendgrouptitle_text="<b><em>O(NP)<br />algorithm<em></b>",
                visible="legendonly",
                showlegend=show_legend,
                line_width=1,
                line_color="#444",
                marker_color=f"rgba{tuple(color[i])}",
                marker_line_color="#444",
                marker_line_width=1,
                fillcolor=f"rgba{tuple(fill[i])}",
                span=[x.min(), x.max()],
                points=False
            ))

            # Add a box plot to have MEAN and SD values.
            plot_distribution.add_trace(go.Box(
                y=x,
                x0=f"{name_label} [O(NP)]</b>",
                name=code,
                legendgroup="ONP",
                legendgrouptitle_text="<b><em>O(NP)<br />algorithm<em></b>",
                visible="legendonly",
                showlegend=False,
                line_width=1.5,
                line_color="#444",
                marker_color=f"rgba{tuple(color[i])}",
                marker_line_color="#444",
                marker_line_width=1,
                fillcolor=f"rgba{tuple(fill[i])}",
                boxmean="sd",
                jitter=0.5
            ))

            # Distribution for the unorder context.
            matrix: np.ndarray = data_list[1][m_i, :][:, m_i]
            x: np.ndarray = matrix[np.triu_indices(matrix.shape[0], k=1)] * 100

            # Those next plot are shown by default.
            # Add a violin plot.
            plot_distribution.add_trace(go.Violin(
                y=x,
                x0=f"{name_label} [BC]",
                name=code,
                legendgroup="BC_dist",
                legendgrouptitle_text=("<b><em>Bray-Curtis<br />"
                                       "distance</em></b>"),
                showlegend=show_legend,
                line_width=1,
                line_color="#444",
                marker_color=f"rgba{tuple(color[i])}",
                marker_line_color="#444",
                marker_line_width=1,
                fillcolor=f"rgba{tuple(fill[i])}",
                span=[x.min(), x.max()],
                points=False
            ))

            # Add a box plot to have MEAN and SD values.
            plot_distribution.add_trace(go.Box(
                y=x,
                x0=f"{name_label} [BC]",
                name=code,
                legendgroup="BC_dist",
                legendgrouptitle_text=("<b><em>Bray-Curtis<br />"
                                       "distance</em></b>"),
                showlegend=False,
                line_width=1.5,
                line_color="#444",
                marker_color=f"rgba{tuple(color[i])}",
                marker_line_color="#444",
                marker_line_width=1,
                fillcolor=f"rgba{tuple(fill[i])}",
                boxmean="sd",
                jitter=0.5
            ))

        # Draw the plot of the network.
        plot: go.FigureWidget = scope_tree.plot_network(
            peitsch_code=code
        )

        save_path: str = join(
            argument["output"],
            f"network_{code}_{date}.html"
        )

        plot.write_html(save_path, full_html=False, include_plotlyjs=True)

    # Add the rectangle border.
    plot_distribution.add_shape(
        type="rect",
        xref="paper",
        yref="paper",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(width=1, color="black")
    )

    plot_distribution.update_traces()

    # Modify general plot properties.
    plot_distribution.update_layout(
        plot_bgcolor="white",
        legend_title="<b>Data distribution</b>",
        margin=dict(l=30, r=30, t=30, b=30),
        font=dict(size=14),
        xaxis_title="<b>Peitsch code</b>",
        yaxis_title="<b>Distance (%)</b>",
        yaxis_tickformat=",.0f",
        boxgroupgap=0.1,
        boxgap=0.1,
        violingroupgap=0.1,
        violingap=0.1
    )

    # Modify axis properties.
    plot_distribution.update_xaxes(showline=True, linewidth=1)
    plot_distribution.update_yaxes(showline=True, linewidth=1)

    # Force to show the legend, even if we have only one trace.
    plot_distribution["data"][0]["showlegend"] = True

    # Save the plot distribution.
    save_path: str = join(
        argument["output"],
        f"context_distribution_{date}.html"
    )

    plot_distribution.write_html(save_path, full_html=False,
                                 include_plotlyjs=True)