"""Display cohorts; the frozen evidence retains every original experiment."""

FACEBOOK = 'ego-Facebook'
MAIN_GRAPHS = ['soc-Epinions1', 'web-Stanford', 'com-Amazon', 'web-Google', 'com-Youtube']
GRAPH_NAMES = {
    FACEBOOK: 'Facebook', 'soc-Epinions1': 'Epinions',
    'web-Stanford': 'Stanford', 'com-Amazon': 'Amazon',
    'web-Google': 'Google', 'com-Youtube': 'YouTube',
}
GRAPH_TYPES = {
    FACEBOOK: 'Social', 'soc-Epinions1': 'Trust',
    'web-Stanford': 'Hyperlinks', 'com-Amazon': 'Products',
    'web-Google': 'Hyperlinks', 'com-Youtube': 'Social',
}


def main_exact(records):
    return [r for r in records if FACEBOOK not in r['dataset']]
