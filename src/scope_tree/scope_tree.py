"""This module contains an object to manipulate SCOPe classification.
"""

__authors__ = ["ROUAUD Lucas"]
__contact__ = ["lucas.rouaud@gmail.com"]
__date__ = "23s/02/2023"
__version__ = "1.0.0"
__copyright__ = "CC BY-SA"


from tqdm import tqdm
from sys import exit as sysexit


class Scope:
    """An object to manipulate SCOPe classification.
    """

    def __init__(self, path: str) -> None:
        """Instantiate a `Scope` object.

        Parameters
        ----------
        path : str
            A path to a SCOPe classification data file.
        """
        self.classification: "dict[str: str]" = {}

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

    def __str__(self) -> str:
        """Change the ` ` function comportment of this object.

        Returns
        -------
        str
            What to print.
        """
        return f"Scope object with {len(self.classification.keys())} items."

    def local_score(self, left_domain: str, right_domain: str) -> int:
        """Compute a score of distance between two SCOPe domains.

        Parameters
        ----------
        left_domain : str
            A domain to compare.
        right_domain : str
            An other domain to compare.

        Returns
        -------
        int
            The compute local score.
        """
        # Check if given domain are in the parse file.
        if left_domain not in self.classification:
            sysexit(f"[Err##] Argument `left_domain` '{left_domain}' is not "
                    "in the parse domain list. Please check what's been given.")
        elif right_domain not in self.classification:
            sysexit(f"[Err##] Argument `right_domain` '{right_domain}' is "
                    "not in the parse domain list. Please check what's been "
                    "given.")

        left_class: "list[str]" = self.classification[left_domain].split(".")
        right_class: "list[str]" = self.classification[right_domain].split(".")

        # Compute the score so the more we are close to the SCOPe classification
        # tree root, the higher the score is.
        for i, left in enumerate(left_class):
            if left != right_class[i]:
                return 4 - i

        return 0

    def global_score(self, domain: "list[str]") -> float:
        """Compute a global score between multiple SCOPe domains. Actually, it
        is the mean between of local score (see `local_score()`) of all pairs of
        domains.

        Parameters
        ----------
        domain : list[str]
            A list containing the SCOPe domains to compare.

        Returns
        -------
        float
            The compute local score.
        """
        length: int = len(domain)

        # You should not have duplicate scores.
        if length != len(set(domain)):
            sysexit(f"[Err##] There is/are {length - len(set(domain))} "
                    "duplicates domains in the given list.")

        score_list: "list[int]" = 0
        count: int = 0

        # Compare all pairs of domains.
        for i, left_domain in enumerate(domain[:-1]):
            if left_domain not in self.classification:
                length -= 1
                continue

            for right_domain in domain[i + 1:]:
                if right_domain not in self.classification:
                    length -= 1
                    continue

                # Sum all local compute scores.
                score_list += self.local_score(left_domain, right_domain)
                count += 1

        # Return the mean of local scores.
        return score_list / count


if __name__ == "__main__":
    scope_manip: object = Scope("data/SCOPe_2.08_classification.txt")

    print(scope_manip.classification["d1dlwa_"])
    print(scope_manip)
    print(scope_manip.local_score("d3ordb_", "d3hx7r2"))
    print(scope_manip.local_score("d3ordb_", "d2gkna_"))
    print(scope_manip.global_score(["d3ordb_", "d3hx7r2", "d2gkna_", "gg"]))
