NO_RELATION = "Other"

LABEL_TO_ID = {'Other': 0,
 'Cause-Effect(e1,e2)': 1,
 'Cause-Effect(e2,e1)': 2,
 'Component-Whole(e1,e2)': 3,
 'Component-Whole(e2,e1)': 4,
 'Content-Container(e1,e2)': 5,
 'Content-Container(e2,e1)': 6,
 'Entity-Destination(e1,e2)': 7,
 'Entity-Destination(e2,e1)': 8,
 'Entity-Origin(e1,e2)': 9,
 'Entity-Origin(e2,e1)': 10,
 'Instrument-Agency(e1,e2)': 11,
 'Instrument-Agency(e2,e1)': 12,
 'Member-Collection(e1,e2)': 13,
 'Member-Collection(e2,e1)': 14,
 'Message-Topic(e1,e2)': 15,
 'Message-Topic(e2,e1)': 16,
 'Product-Producer(e1,e2)': 17,
 'Product-Producer(e2,e1)': 18}

ID_TO_LABEL = dict([(val, key) for key, val in LABEL_TO_ID.items()])

GRAMMER_NER_START = ["<e1>", "<e2>"]
GRAMMER_NER_END = ["</e1>", "</e2>"]