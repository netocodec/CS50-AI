import os
import random
import re
import sys
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    result = {}
    total_pages = len(corpus)
    current_page_links = corpus[page]
    total_linked_pages = len(current_page_links)
    if current_page_links:
        for page_key in corpus:
            result[page_key] = (1-damping_factor)/total_pages

        for page_key in corpus[page]:
            result[page_key] += damping_factor/total_linked_pages
    else:
        for page_key in corpus:
            result[page_key] = 1/total_pages

    return result


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages_dic = {}.fromkeys(corpus.keys(), 0)
    page = random.choice(list(pages_dic.items()))[0]

    for page_key in range(1, n):
        page_distribution = transition_model(corpus, page, damping_factor)
        for next_page_key in pages_dic:
            pages_dic[next_page_key] = (((page_key-1) * pages_dic[next_page_key]) + page_distribution[next_page_key]) / page_key
        page = random.choices(list(pages_dic.keys()), list(pages_dic.values()), k=1)[0]

    return pages_dic


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    total_pages = len(corpus)
    pages_dic = {}.fromkeys(corpus.keys(), 1/total_pages)

    change = True
    while change:
        change = False
        old_pages = copy.deepcopy(pages_dic)

        for page_key in corpus:
            pages_dic[page_key] = ((1 - damping_factor)/total_pages) + (damping_factor * get_sum(corpus, pages_dic, page_key))
            change = change or abs(old_pages[page_key] - pages_dic[page_key]) > 0.001

    return pages_dic

def get_sum(corpus, pages_dic, page):
    result = 0
    for page_key in corpus:
        if page in corpus[page_key]:
            result += pages_dic[page_key] / len(corpus[page_key])

    return result

if __name__ == "__main__":
    main()
