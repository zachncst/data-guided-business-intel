import twitterStream
import unittest

class TestTwitterStreamMethods(unittest.TestCase):

    def test_load_wordlist(self):
        pwords = twitterStream.load_wordlist("positive.txt")
        self.assertIsNotNone(pwords)
        self.assertIn("reform", pwords)

    def testGraph(self):
        counts = [   [],
        [('positive', 255), ('negative', 101)],
        [('positive', 234), ('negative', 133)],
        [('positive', 235), ('negative', 142)],
        [('positive', 221), ('negative', 142)],
        [('positive', 235), ('negative', 107)],
        [('positive', 208), ('negative', 120)]]

        twitterStream.make_plot(counts)

     
    if __name__ == '__main__':
        unittest.main()
