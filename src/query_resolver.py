from src.utilities.global_timers import timeit, timers
import numpy

actions = ["|", "_", "^", "&"]  # union  # diff  # symetric diff  # intersection
bad_chars = ["=", "==", "?", "+", "!", "<", ">", "#", "@", "%", "*", "$"]
min_num_valid_query_parts = 3 # just a player query

class QueryNode:
    def __init__(self, data= None):
        self.query_parts = data # part of query string
        self.left = None
        self.right = None
        self.parent = None
        self.units = set() # set of indices/stat calculations
    
    def __getitem__(self,key):
        if key == "query_parts":
            return self.query_parts
        elif key == "units":
            return self.units
        else:
            raise Exception("No key {k}".format(k=key))

    def resolve_units(self, season, aggregator, debug=False):
        try:
            if self.left:
                self.left.resolve_units(season, aggregator, debug)
            if self.right:
                self.right.resolve_units(season, aggregator, debug)
            if self.query_parts not in actions: # so a player
                self.units = self.grab_possessions(self.query_parts, season, aggregator, debug)
            else:
                self.units = self.perform_set_operation(
                    self.left, self.right, self.query_parts, debug
                )
        except Exception as error:
            print(error)
            raise

    def perform_set_operation(self, l_operand, r_operand, operator, debug=False):
        set_result = set()
        l_units = numpy.array(l_operand.units)
        r_units = numpy.array(r_operand.units)

        if operator not in actions:
            raise Exception("Invalid operator", operator)
        if operator == "|":
            if debug:
                print(
                    "Performing union on {l} and {r}".format(
                        l=l_operand.query_parts, r=r_operand.query_parts
                    )
                )
            set_result = numpy.union1d(l_units, r_units)
        if operator == "_":
            if debug:
                print(
                    "Performing diff on {l} and {r}".format(
                        l=l_operand.query_parts, r=r_operand.query_parts
                    )
                )
            set_result = numpy.setdiff1d(l_units, r_units)
        if operator == "&":
            if debug:
                print(
                    "Performing intersection on {l} and {r}".format(
                        l=l_operand.query_parts, r=r_operand.query_parts
                    )
                )
            set_result = numpy.intersect1d(l_units, r_units)
        if operator == "^":
            if debug:
                print(
                    "Performing symmetric diff on {l} and {r}".format(
                        l=l_operand.query_parts, r=r_operand.query_parts
                    )
                )
            set_result = numpy.setxor1d(l_units, r_units)

        self.query_parts = list(
            (l_operand.query_parts, operator, r_operand.query_parts)
        )  # combined for debug readouts
        return set_result

    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.query_parts)
        if self.right:
            self.right.print_tree()

    def grab_possessions(self, player, season, aggregator, debug=False):
        return set(aggregator[player, season][1])  # this is going to be a db read


class ExpressionTree:
    def __init__(self):
        self.root = QueryNode() # contains set of units for whole query after resolved
        self.size = 1
        self.curr = self.root

    def insert(self, data):
        try:
            if data == "(":
                self.curr.left = QueryNode()
                self.curr.left.parent = self.curr
                if self.size == 1:
                    self.root = self.curr
                self.curr = self.curr.left
                self.size += 1
            if data in actions:
                self.curr.query_parts = data
                self.curr.right = QueryNode()
                self.curr.right.parent = self.curr
                self.curr = self.curr.right
                self.size += 1
            if data not in actions + ["(", ")"]: # so player
                self.curr.query_parts = data
                self.curr = self.curr.parent
            if data == ")":
                self.curr = self.curr.parent
        except AttributeError as error:
            print("error when building ExpressionTree",error)
            raise

    def resolve_units(
        self, season, aggregator, debug=False
    ):  # eventually will swap agg with DB
        self.root.resolve_units(season, aggregator, debug)
        return self.root.units

    def print_tree(self):
        self.root.print_tree()

    def print_size(self):
        print(self.size)


class QueryResolver:
    def __init__(self):
        self.parse_tree = ExpressionTree()

    def __getitem__(self, key):
        if key == "parse_tree":
            return self.parse_tree
        else:
            raise Exception("No key {k}".format(k=key))

    @timeit
    def resolve_query(
        self, query, season, aggregator, debug=False
    ):  
        if debug:
            print(query)

        query = self.sanitize_query(query, debug)

        self.build_expression_tree(query.split(" "),debug)
        self.resolve_units(season, aggregator, debug)

    def build_expression_tree(self, query_parts, debug=False):
        tree = ExpressionTree()
        base_case = len(query_parts) == min_num_valid_query_parts

        if base_case:
            single_player_query = query_parts[1]
            tree.insert(single_player_query)
        else:
            for part in query_parts:
                tree.insert(part)
        self.parse_tree = tree

        if debug:
            print(query_parts)
            self.parse_tree.print_size()
            self.parse_tree.print_tree()

    def resolve_units(self, season, aggregator, debug=False):
        return self.parse_tree.resolve_units(season, aggregator, debug)

    def sanitize_query(self, query, debug=False):
        bad_chars = []
        char_idx = 0
        open_perens,closed_perens = 0,0
        query = query.strip()
        query_len = len(query)

        # handle bad spacing within query. Jank city. Find something simpler
        try:
            while char_idx < query_len:
                if query[char_idx].isdigit() or query[char_idx] in bad_chars:
                    bad_chars.append(query[char_idx])
                    break
                if query[char_idx] in actions:
                    if query[char_idx + 1] != " ":
                        query = query[: char_idx + 1] + " " + query[char_idx + 1 :]
                        query_len = len(query)
                    if query[char_idx - 1] != " ":
                        query = query[:char_idx] + " " + query[char_idx:]
                        query_len = len(query)
                        char_idx += 1
                if query[char_idx] == "(":
                    open_perens += 1
                    if query[char_idx + 1] != " ":
                        query = query[: char_idx + 1] + " " + query[char_idx + 1 :]
                        query_len = len(query)
                    if query[char_idx - 1] != " " and char_idx != 0:
                        query = query[:char_idx] + " " + query[char_idx:]
                        query_len = len(query)
                        char_idx += 1
                if query[char_idx] == ")":
                    closed_perens += 1
                    if query[char_idx - 1] != " ":
                        query = query[:char_idx] + " " + query[char_idx:]
                        query_len = len(query)
                        char_idx += 1
                char_idx += 1
        except IndexError as error:
            print("error query santizer: {e}".format(e=error))

        if bad_chars:
            raise Exception("Bad query {q}, {b} found".format(q=query, b=bad_chars))

        # handle lack of open or close parens
        if query[0] != "(" and query[-1] != ")":
            query = "( {0} )".format(query)
        elif query[0] != "(" and query[-1] == ")":
            if open_perens == closed_perens:
                query = "( {0} )".format(query)
            else:
                query = "( {0}".format(query)
                open_perens += 1
        elif query[0] == "(" and query[-1] != ")":
            if open_perens == closed_perens:
                query = "( {0} )".format(query)
            else:
                query = "{0} )".format(query)
                closed_perens += 1

        if open_perens != closed_perens:
            raise Exception(
                "Could not parse query {q}, are you missing a '(' or ') ?".format(
                    q=query
                )
            )
        if debug:
            print("sanitized query:", query)

        return query


if __name__ == "__main__":
    import h5py
    from src.data.batch_loader import load_raw_data
    from src.index_finder import IndexFinder
    import os

    db_path = "cache/batch_loader_unit_test.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    ag = IndexFinder()
    with h5py.File(db_path, "a") as db:
        load_raw_data(db, years=[2018], season_types=["Playoffs"])
        season = "2018_playoffs"
        ag.aggregate(season, db)

    
    # required to have parens explicit, rather than relying on left to right operation, see test 7
    test1 = "( lebron-james | ( stephen-curry | ( draymond-green | andre-iguodala ) ) )"
    test2 = "( lebron-james | ( stephen-curry ^ ( draymond-green | andre-iguodala ) ) )"
    test3 = "( ( kevin-durant | stephen-curry ) ^ ( lebron-james | kyle-korver ) )"
    test4 = "( draymond-green | andre-iguodala ) "
    test5 = "( ( ( draymond-green & andre-iguodala ) _ stephen-curry ) & kyle-korver )"
    test6 = "( ( stephen-curry | draymond-green ) ^ lebron-james )"
    test7 = "( kevin-love | ( stephen-curry ^ kevin-durant ) )"  # note: "( A | ( B ^ C ) | D )" would be wrong, parens needed
    test8 = "( ( lebron-james & ( stephen-curry | kevin-durant ) ) & kevin-love )"
    test9 = "( ( ( lebron-james & kevin-love ) | ( kevin-durant & stephen-curry ) ) | draymond-green )"
    # sanitation tests
    test10 = "(draymond-green|andre-iguodala)"
    test11 = "draymond-green|andre-iguodala"
    test12 = "(draymond-green|andre-iguodala"
    test13 = "draymond-green | andre-iguodala)"
    test14 = " lebron-james|(stephen-curry|(draymond-green|andre-iguodala)) "
    test15 = " (lebron-james |stephen-curry) "
    test16 = (
        "(((lebron-james&kevin-love )| (kevin-durant &stephen-curry) )| draymond-green)"
    )
    test17 = "lebron-james | stephen-curry"
    # base case test
    test18 = "( stephen-curry )"

    qr = QueryResolver()
    debug = True
    
    print("Test 1")
    qr.resolve_query(test1, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 2")
    qr.resolve_query(test2, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 3")
    qr.resolve_query(test3, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 4")
    qr.resolve_query(test4, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 5")
    qr.resolve_query(test5, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 6 ")
    qr.resolve_query(test6, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 7")
    qr.resolve_query(test7, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 8")
    qr.resolve_query(test8, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 9")
    qr.resolve_query(test9, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 10")
    qr.resolve_query(test10, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 11")
    qr.resolve_query(test11, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 12")
    qr.resolve_query(test12, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 13")
    qr.resolve_query(test13, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 14")
    qr.resolve_query(test14, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 15")
    qr.resolve_query(test15, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 16")
    qr.resolve_query(test16, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    print("Test 17")
    qr.resolve_query(test17, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()
    
    print("Test 18")
    qr.resolve_query(test18, season, ag, debug)
    print(qr["parse_tree"].root["query_parts"])
    print()
    print()
    print()

    # only care about the resolve_query time
    # should be sub .0008 per call
    # turn debug off for timing tests
    print(timers.total())
    print(timers.avg())
   



    