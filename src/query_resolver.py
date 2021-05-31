
#import numpy

actions = [
    '|',
    '_',
    '^',
    '&'
]

class Node: 
    def __init__(self,data):
        self.data = data
        self.left = None
        self.right = None
        self.parent = None
        self.units = set()
    
    def resolve_units(self,season,aggregator):
        if self.left:
            self.left.resolve_units(season,aggregator)
        if self.right:
            self.right.resolve_units(season,aggregator)
        if self.data not in actions:
            self.units = self.grab_possessions(self.data,season,aggregator)
        else:
            self.units = self.perform_set_operation(self.left,
                self.right,self.data,True)
        
    
    def perform_set_operation(self,l_operand,r_operand,operator,debug = False):
        set_result = set()
        l_units = l_operand.units
        r_units = r_operand.units
        
        # Note this needs to be numpy set funcs() once you get this working, not python funcs
        if operator not in actions:
            raise Exception("Invalid operator",operator)
        if operator == '|':
            if debug:
                print("Performing union on {l} and {r}".format(l = l_operand.data, r = r_operand.data))
            set_result = set.union(l_units, r_units)
        if operator == '_':
            if debug:
                print("Performing diff on {l} and {r}".format(l = l_operand.data, r = r_operand.data))
            set_result = set.difference(l_units, r_units)
        if operator == '&':
            if debug:
                print("Performing intersection on {l} and {r}".format(l = l_operand.data, r = r_operand.data))
            set_result = set.intersection(l_units, r_units)
        if operator == '^':
            if debug:
                print("Performing symmetric diff on {l} and {r}".format(l = l_operand.data, r = r_operand.data))
            set_result = set.symmetric_difference(l_units, r_units)
    
        self.data = list((l_operand.data, operator, r_operand.data)) # combined for debug readouts
        return set_result
    
    def print_tree(self):
        if self.left:
            self.left.print_tree()
        print(self.data)
        if self.right:
            self.right.print_tree()

    def grab_possessions(self,player,season,aggregator):
        return set(aggregator[player,season][1]) # this is going to be a db read


class ExpressionTree:
    def __init__(self):
        self.root = Node("blank")
        self.size = 1
        self.this = self.root


    def insert(self,data):
        if data == '(':
            self.this.left = Node("blank")
            self.this.left.parent = self.this
            if self.size == 1:
                self.root = self.this
            self.this = self.this.left
            self.size += 1
        if data in actions:
            self.this.data = data
            self.this.right = Node("blank")
            self.this.right.parent = self.this
            self.this = self.this.right
            self.size +=1
        if data not in actions and data not in ['(',')']:
            self.this.data = data
            self.this = self.this.parent
        if data == ')':
            self.this = self.this.parent 
    
    def print_tree(self):
        self.root.print_tree()

    def resolve_units(self,season,aggregator): # eventually will swap agg with DB
        self.root.resolve_units(season,aggregator)
        return self.root.units
    
    def print_size(self):
        print(self.size)


class QueryResolver:
    def __init__(self) -> None:
        self.groups = []
        self.parse_tree = ExpressionTree()

    def __getitem__(self, key):
        if key == "groups":
            return self.groups
        elif key =="parse_tree":
            return self.parse_tree
        else:
            raise Exception("No key {k}".format(k = key))

    def resolve_units(self,season,aggregator):
        return self.parse_tree.resolve_units(season,aggregator)


    def grouper(self,query,debug = False): # use append and pop
        query = query.strip()

        if self.groups:
            self.groups.clear()

        # some gary bullshit
        open_paren_stack = []
        close_peren_queue = []
        for idx in range(0,len(query)):
            char = query[idx]
            if char != '(' and char != ')':
                continue
            elif char == '(':
                open_paren_stack.append((idx))
            else :
                close_peren_queue.append(idx)
                if len(open_paren_stack) > 0:
                    start,end = (open_paren_stack.pop(),close_peren_queue.pop(0))
                    if start == 0:
                        group = query[start:end+1]
                        if group != "":
                            self.groups.append(group)
        
        self.get_action_groups(debug)


    def get_action_groups(self,debug = False):
        groups = self.groups
               
        self.build_expression_tree(groups[0].split(" "))

        if(debug):
            self.parse_tree.print_size()
            self.parse_tree.print_tree()


    def build_expression_tree(self,group_parts):
        print(group_parts)
        tree = ExpressionTree()

        for part in group_parts:
            tree.insert(part)
        self.parse_tree = tree


if __name__ == "__main__":
    import h5py
    from src.data.batch_loader import load_raw_data
    from src.aggregator import Aggregator
    import os

    db_path = "cache/batch_loader_unit_test.h5"
    if os.path.exists(db_path):
        os.remove(db_path)

    qr = QueryResolver()
    ag  = Aggregator()
    with h5py.File(db_path, "a") as db:
        load_raw_data(db, years=[2018], season_types=["Playoffs"])
        season = "2018_playoffs"
        ag.aggregate(season,db)
        
    
    # sanity check
    games,pos,plays = ag["andre-iguodala",season]
    print(len(pos))
    debug = True
    # required to have a space between chars and for at least one set of parens. 
    # also required to have parens explicit, rather than relying on left to right operation, see test 7
     
    test1 = "( lebron-james | ( stephen-curry | ( draymond-green | andre-iguodala ) ) )"
    test2 = "( lebron-james | ( stephen-curry ^ ( draymond-green | andre-iguodala ) ) )"
    test3 = "( ( kevin-durant | stephen-curry ) ^ ( lebron-james | kyle-korver ) )"
    test4 = "( draymond-green | andre-iguodala ) "
    test5 = "( ( draymond-green & andre-iguodala ) _ ( lebron-james & kyle-korver ) )"
    test6 = "( ( stephen-curry | draymond-green ) ^ ( lebron-james _ kevin-love ) )"
    test7 = "( ( kevin-love | ( stephen-curry ^ kevin-durant ) ) | kyle-korver )" # note: "( A | ( B ^ C ) | D )" would be wrong, parens needed

    print("Test 1")
    qr.grouper(test1,debug)

    qr.resolve_units(season,ag)

    ## refactor aggregator passing 
    ## f-ton of unit tests


    print()
    print()
    print()

    print("Test 2")
    qr.grouper(test2,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()
    
    print("Test 3")
    qr.grouper(test3,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()

    print("Test 4")
    qr.grouper(test4,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()

    print("Test 5")
    qr.grouper(test5,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()

    print("Test 6 ")
    qr.grouper(test6,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()

    print("Test 7")
    qr.grouper(test7,debug)
    qr.resolve_units(season,ag)
    print()
    print()
    print()

