import unittest


class TestQResolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # functional tests
        cls.test1 = "( lebron-james | ( stephen-curry | ( draymond-green | andre-iguodala ) ) )"
        cls.test2 = "( ( kevin-durant | stephen-curry ) ^ ( lebron-james | kyle-korver ) )"
        cls.test3 = "( draymond-green | andre-iguodala ) "
        cls.test4 = "( ( ( draymond-green & andre-iguodala ) _ stephen-curry ) & kyle-korver )"
        cls.test5 = "( ( stephen-curry | draymond-green ) ^ lebron-james )"
        cls.test6 = "( kevin-love | ( stephen-curry ^ kevin-durant ) )"  # note: "( A | ( B ^ C ) | D )" would be wrong, parens needed
        cls.test7 = "( ( lebron-james & ( stephen-curry | kevin-durant ) ) & kevin-love )"
        cls.test8 = "( ( ( lebron-james & kevin-love ) | ( kevin-durant & stephen-curry ) ) | draymond-green )"
        # sanitation tests
        cls.test9 = "(draymond-green|andre-iguodala)"
        cls.test10 = "draymond-green|andre-iguodala"
        cls.test11 = "(draymond-green|andre-iguodala"
        cls.test12 = "draymond-green | andre-iguodala)"
        cls.test13 = " lebron-james|(stephen-curry|(draymond-green|andre-iguodala)) "
        cls.test14 = " (lebron-james |stephen-curry) "
        cls.test15 = (
        "(((lebron-james&kevin-love )| (kevin-durant &stephen-curry) )| draymond-green)"
        )
        cls.test16 = "lebron-james | stephen-curry"
        # base case test
        cls.test17 = "( stephen-curry )"
        #expected responses
        cls.test1_answer = ['lebron-james', '|', ['stephen-curry', '|', ['draymond-green', '|', 'andre-iguodala']]]
        cls.test2_answer = [['kevin-durant', '|', 'stephen-curry'], '^', ['lebron-james', '|', 'kyle-korver']]
        cls.test3_answer = ['draymond-green', '|', 'andre-iguodala']
        cls.test4_answer = [[['draymond-green', '&', 'andre-iguodala'], '_', 'stephen-curry'], '&', 'kyle-korver']
        cls.test5_answer = [['stephen-curry', '|', 'draymond-green'], '^', 'lebron-james']
        cls.test6_answer = ['kevin-love', '|', ['stephen-curry', '^', 'kevin-durant']]
        cls.test7_answer = [['lebron-james', '&', ['stephen-curry', '|', 'kevin-durant']], '&', 'kevin-love']
        cls.test8_answer = [[['lebron-james', '&', 'kevin-love'], '|', ['kevin-durant', '&', 'stephen-curry']], '|', 'draymond-green']
        # sanitation tests
        cls.test9_answer = ['draymond-green', '|', 'andre-iguodala']
        cls.test10_answer = ['draymond-green', '|', 'andre-iguodala']
        cls.test11_answer = ['draymond-green', '|', 'andre-iguodala']
        cls.test12_answer = ['draymond-green', '|', 'andre-iguodala']
        cls.test13_answer = ['lebron-james', '|', ['stephen-curry', '|', ['draymond-green', '|', 'andre-iguodala']]]
        cls.test14_answer = ['lebron-james', '|', 'stephen-curry']
        cls.test15_answer = [[['lebron-james', '&', 'kevin-love'], '|', ['kevin-durant', '&', 'stephen-curry']], '|', 'draymond-green']
        cls.test16_answer = ['lebron-james', '|', 'stephen-curry']
        #base case
        cls.test17_answer = "stephen-curry"
        # timeing test
        cls.time_to_beat = 0.00075
        # resolver
        cls.qr = QueryResolver()

        cls.ag = IndexFinder()

        db_path = "cache/batch_loader_unit_test.h5"
        if os.path.exists(db_path):
            os.remove(db_path)
        with h5py.File(db_path, "a") as db:
            load_raw_data(db, years=[2018], season_types=["Playoffs"])
            cls.season = "2018_playoffs"
            cls.ag.aggregate(cls.season, db)

    def test_1(self):
        self.qr.resolve_query(self.test1, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test1_answer)

    def test_2(self):
        self.qr.resolve_query(self.test2, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test2_answer)

    def test_3(self):
        self.qr.resolve_query(self.test3, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test3_answer)
    
    def test_4(self):
        self.qr.resolve_query(self.test4, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test4_answer)
    
    def test_5(self):
        self.qr.resolve_query(self.test5, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test5_answer)
    
    def test_6(self):
        self.qr.resolve_query(self.test6, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test6_answer)
    
    def test_7(self):
        self.qr.resolve_query(self.test7, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test7_answer)
    
    def test_8(self):
        self.qr.resolve_query(self.test8, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test8_answer)
    
    def test_9(self):
        self.qr.resolve_query(self.test9, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test9_answer)
    
    def test_10(self):
        self.qr.resolve_query(self.test10, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test10_answer)
    
    def test_11(self):
        self.qr.resolve_query(self.test11, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test11_answer)
    
    def test_12(self):
        self.qr.resolve_query(self.test12, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test12_answer)
    
    def test_13(self):
        self.qr.resolve_query(self.test13, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test13_answer)
    
    def test_14(self):
        self.qr.resolve_query(self.test14, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test14_answer)
    
    def test_15(self):
        self.qr.resolve_query(self.test15, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test15_answer)
    
    def test_16(self):
        self.qr.resolve_query(self.test16, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test16_answer)
    
    def test_17(self):
        self.qr.resolve_query(self.test17, self.season, self.ag)
        result = self.qr["parse_tree"].root["query_parts"]
        self.assertEqual(result, self.test17_answer)

    def sanity_check(self):
        self.assertFalse(self.test2_answer == self.test1_answer)
    
    def test_time(self):
        t = timers.avg()
        start,end = t.find("resolve_query"),len(t)
        query_time_parts = t[start:end].split(" ")
        result = float(query_time_parts[1])
        print("per call runtime:",result)
        self.assertTrue(result < self.time_to_beat)



if __name__ == '__main__':
    import h5py
    from src.utilities.global_timers import timers
    from src.data.batch_loader import load_raw_data
    from src.query_resolver import QueryResolver
    from src.index_finder import IndexFinder
    import os

    unittest.main()