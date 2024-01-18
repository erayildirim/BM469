import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.tag import pos_tag

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def is_content_word(word):
    # Filter out stop words and non-content words
    return word.isalnum() and word.lower() not in stopwords.words('english') and wordnet.synsets(word)

def calculate_fkgl(text):
    words = word_tokenize(text)
    words = [word.lower() for word in words if is_content_word(word)]
    
    num_words = len(words)
    num_sentences = text.count('.') + text.count('!') + text.count('?')
    
    if num_words == 0 or num_sentences == 0:
        return float('inf')
    
    syllable_count = sum([syllables(word) for word in words])
    
    fkgl = 0.39 * (num_words / num_sentences) + 11.8 * (syllable_count / num_words) - 15.59
    return fkgl

def syllables(word):
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count

def crossover(parent1, parent2):
    point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = get_synonym(individual[i])
    return individual

def get_pos(word):
    tag = pos_tag([word])[0][1]
    if tag.startswith('N'):
        return 'n'  # Noun
    elif tag.startswith('V'):
        return 'v'  # Verb
    elif tag.startswith('R'):
        return 'r'  # Adverb
    elif tag.startswith('J'):
        return 'a'  # Adjective
    else:
        return ''

def get_synonym(word):
    pos = get_pos(word)

    if pos:
        synsets = wordnet.synsets(word, pos=pos)
        lemmas = [lemma.name() for synset in synsets for lemma in synset.lemmas()]
        lemmas = [lemma for lemma in lemmas if lemma.isalpha() and lemma.lower() != word.lower()]

        # Avoid replacing common words and certain types
        if not is_content_word(word) or word.lower() in ["in", "on", "at", "and", "the", "is", "was", "it"]:
            return word

        if lemmas:
            return random.choice(lemmas)

    return word


def optimize_fkgl(input_text, population_size=20, generations=100, mutation_rate=0.01):
    population = [list(word_tokenize(input_text.lower())) for _ in range(population_size)]

    for generation in range(generations):
        scores = [(calculate_fkgl(TreebankWordDetokenizer().detokenize(individual)), individual) for individual in population]
        scores.sort()

        if generation % 10 == 0:
            print(f"Generation {generation}: Best FKGL Score = {scores[0][0]}")

        # Select top 20% for reproduction
        selected_parents = [individual for _, individual in scores[:population_size // 5]]

        # Crossover
        offspring = []
        for i in range(0, len(selected_parents), 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # Mutation
        offspring = [mutate(individual, mutation_rate) for individual in offspring]

        # Replace old population with new population
        population = selected_parents + offspring

    best_individual = scores[0][1]
    optimized_text = TreebankWordDetokenizer().detokenize(best_individual)
    return optimized_text

if __name__ == "__main__":
    input_text = "The sea moderates the climate and has important roles in the water cycle, carbon cycle, and nitrogen cycle. Humans harnessing and studying the sea have been recorded since ancient times, and evidenced well into prehistory, while its modern scientific study is called oceanography. The most abundant solid dissolved in seawater is sodium chloride. The water also contains salts of magnesium, calcium, potassium, and mercury, amongst many other elements, some in minute concentrations. Salinity varies widely, being lower near the surface and the mouths of large rivers and higher in the depths of the ocean; however, the relative proportions of dissolved salts vary little across the oceans."
    optimized_text = optimize_fkgl(input_text)
    print("\nOriginal Text:\n", input_text)
    print("\nOptimized Text:\n", optimized_text)
