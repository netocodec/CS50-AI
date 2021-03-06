import itertools
import random


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        mines_result = set()
        if len(self.cells) == self.count:
            mines_result = set(self.cells)

        return mines_result

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        safes_result = set()
        if self.count == 0:
            safes_result = set(self.cells)

        return safes_result

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.
        """
        if cell in self.cells:
            self.count -= 1
            self.cells.remove(cell)
            return 1
        return 0

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            return 1
        return 0


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        cell_counter = 0
        self.mines.add(cell)
        for sentence in self.knowledge:
            cell_counter += sentence.mark_mine(cell)

        return cell_counter

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        cell_counter = 0
        self.safes.add(cell)
        for sentence in self.knowledge:
            cell_counter += sentence.mark_safe(cell)

        return cell_counter

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        self.moves_made.add(cell)
        self.mark_safe(cell)

        neighbours = self.get_neighbours(cell)
        new_sentence = Sentence(neighbours, count)
        for mine in self.mines:
            new_sentence.mark_mine(mine)
        for safe in self.safes:
            new_sentence.mark_safe(safe)

        self.knowledge.append(new_sentence)
        self.update_sentences()

        new_sentences = self.get_new_sentences()
        while new_sentences:
            for sentence in new_sentences:
                self.knowledge.append(sentence)

            self.update_sentences()
            new_sentences = self.get_new_sentences()


    def get_new_sentences(self):
        new_sentences = []

        for current_sentence, next_sentence in itertools.combinations(self.knowledge, 2):
            new_created_sentence = None

            if next_sentence.cells.issubset(current_sentence.cells):
                new_created_sentence = Sentence(current_sentence.cells - next_sentence.cells, current_sentence.count - next_sentence.count)
            elif current_sentence.cells.issubset(next_sentence.cells):
                new_created_sentence = Sentence(next_sentence.cells - current_sentence.cells, next_sentence.count - current_sentence.count)

            if new_created_sentence is not None and new_created_sentence not in self.knowledge:
                new_sentences.append(new_created_sentence)

        for sentence in self.knowledge:
            if sentence.cells is set():
                self.knowledge.remove(sentence)

        return new_sentences

    def update_sentences(self):
        cell_counter = 1

        while cell_counter:
            cell_counter = 0
            for sentence in self.knowledge:
                for cell in sentence.known_safes():
                    self.mark_safe(cell)
                    cell_counter += 1
                for cell in sentence.known_mines():
                    self.mark_mine(cell)
                    cell_counter += 1

            for cell in self.safes:
                cell_counter += self.mark_safe(cell)
            for cell in self.mines:
                cell_counter += self.mark_mine(cell)

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for safe in self.safes:
            if safe not in self.moves_made and safe not in self.mines:
                return safe
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        for row in range(self.width):
            for col in range(self.height):
                cell = (row, col)
                if cell not in self.moves_made and cell not in self.mines:
                    return cell
        return None

    def get_neighbours(self, cell):
        current_cell_row, current_cell_col = cell
        neighbors_result = set()

        for row in range(max(0, current_cell_row-1), min(current_cell_row+2, self.height)):
            for col in range(max(0, current_cell_col-1), min(current_cell_col+2, self.width)):
                if (row, col) != (current_cell_row, current_cell_col):
                    neighbors_result.add((row, col))

        return neighbors_result
