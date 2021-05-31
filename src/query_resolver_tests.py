
import unittest

# query parsing tests

class TestQueryParsing(unittest.TestCase):
    def setUp(self):
        from src.query_resolver import QueryResolver, actions

        self.query_resolver = QueryResolver()
        self.debug = False

        self.left_combine_right_compound_test = "( A | ( B | ( C | D ) ) )"
        self.right_combine_left_compound_test = "( ( ( A ^ B ) & C ) & D )"
        self.left_right_combine_test = "( ( A | B ) ^ ( C _ D ) )"
        self.base_test = "( A & B ) "
         # note: "( A | ( B ^ C ) | D )" would be wrong, parens needed
        self.inner_left_right_test = "( ( A | ( B ^ C ) ) | D )"
        self.left_right_test = "( ( A & B ) & C )"


    def test_left_combine_right_compound_test(self):
        self.query_resolver.grouper(self.left_combine_right_compound_test)

        tree = self.query_resolver["parse_tree"]
        result = tree.root.data
        expected = result
        print(tree.root.print_tree())
        self.assertEqual(expected, result)


# possession/play resolver tests

class TestPossessionResolver(unittest.TestCase):
    def setUp(self):
        import h5py
        from src.data.batch_loader import load_raw_data
        from src.aggregator import Aggregator
        import os
        self.aggregator = Aggregator()

        db_path = "cache/batch_loader_unit_test.h5"
        if os.path.exists(db_path):
            os.remove(db_path)
        
        with h5py.File(db_path, "a") as db:
            load_raw_data(db, years=[2018], season_types=["Playoffs"])
            self.season = "2018_playoffs"
            self.aggregator.aggregate(self.season,db)

# stats resolver tests






if __name__ == "__main__":
    unittest.main()