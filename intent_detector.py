AGGREGATIONS = {

    "total":"SUM",

    "sum":"SUM",

    "average":"AVG",

    "avg":"AVG",

    "minimum":"MIN",

    "min":"MIN",

    "maximum":"MAX",

    "max":"MAX",

    "count":"COUNT"

}

class IntentDetector:

    def detect_aggregation(self, tokens):

        for token in tokens:

            if token in AGGREGATIONS:

                return AGGREGATIONS[token]

        return None