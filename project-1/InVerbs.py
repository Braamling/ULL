from collections import Counter

class InVerbs:

    def __init__(self, config):
        self.config = config

        # Load all data from the configured data file
        with open(self.config.data) as data:
            verbs = {}
            for pair in data:
                # Somehow each line has trailing whitespaces
                pair = pair.rstrip()

                # split verb and noun
                verb, noun = pair.split(" ", 1)

                if "_s_" in verb:
                    # Set dict item for verb and noun count
                    if verb in verbs:
                        verbs[verb].append(noun)
                    else:
                        verbs[verb] = [noun]

        in_verbs = []
        for key in verbs.keys():
            counts = dict(Counter(verbs[key]))
            in_verbs.append( (key, counts, len(verbs[key])))

        self.verbs = sorted(in_verbs, key=lambda x: 
                            -x[2])[:self.config.ints_verb_count]


    def get_verbs(self):
        return self.verbs