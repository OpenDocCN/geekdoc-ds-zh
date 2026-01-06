# 稳定婚姻问题

> [`www.algorithm-archive.org/contents/stable_marriage_problem/stable_marriage_problem.html`](https://www.algorithm-archive.org/contents/stable_marriage_problem/stable_marriage_problem.html)

想象你有两组人，每组的大小为 n。每组中的每个人都与对方组中的所有成员有一个内部排名。*稳定匹配问题*试图将两组人联合成稳定的配对。在这种情况下，如果不存在任何一对比他们当前伴侣更喜欢彼此的配对，则认为一组配对是稳定的。这并不意味着每个人都得到了他们的首选，但如果一个人更喜欢另一个人，而那个人也喜欢他们，那么这组配对就不是稳定的。

现在，这通常被讲述成一个故事。一组是男性，另一组是女性，每个人都结婚了，因此得名*稳定婚姻问题*。这个问题通过 Gale-Shapley 算法解决，可以简单地描述如下：

1.  所有男性都向他们首选的女性求婚。

1.  女性会暂时与她们首选的求婚男性订婚。

1.  所有被拒绝的男性都会向他们的下一个选择求婚，而女性则再次选择她们偏好的男性，可能拒绝她们已经订婚的那个人。

这个过程会一直持续到所有个体都被配对，这意味着这个算法保证了稳定的匹配，并且具有一个运行时间。为了清楚起见，尽管这个算法在概念上很简单，但正确实现它相当棘手。我绝不声称这里提供的代码是高效的，我们将来在有了更多工具后一定会回到这个问题上。我非常感兴趣地想看看你们会做什么，以及你们如何实现这个算法。

## 视频解释

下面是一个描述稳定婚姻问题的视频：

[`www.youtube-nocookie.com/embed/A7xRZQAQU8s`](https://www.youtube-nocookie.com/embed/A7xRZQAQU8s)

## 示例代码

```
class Person
    def initialize(id, name, prefs)
        @id      = id
        @name    = name
        @prefs   = prefs
        @partner = nil
        @choices = 0
    end

    def lonely?
        @partner.nil?
    end

    def propose(partners)
        unless self.lonely?
            raise '%s is not lonely!' % self.name
        end
        choice = @prefs[@choices]
        partners[choice].onPropose(self)
        @choices += 1
    end

    def to_s
      "#{@name.rjust(20)}: #{self.lonely? && "Lonely" || @partner.name}"
    end

    def self.generate(size, prefix, r)
        Array.new(size){|i|
            Person.new(
                i,
                "#{prefix}  #{i}",
                (0 ... size).to_a.shuffle(random: r)
            )
        }
    end

    protected
    attr_reader :id, :name
    attr_writer :partner

    # Acts upon a given Proposal
    def onPropose(partner)
        unless self.lonely?
            offer = score(partner)
            current = score(@partner)
            return unless offer > current 
            @partner.partner = nil
        end
        @partner = partner
        partner.partner = self
    end

    private
    # Determines the preference of a given partner
    def score(partner)
        return 0 if partner.nil?
        @prefs.size - @prefs.index(partner.id)
    end
end

# Deterministic Output, feel free to change seed
r = Random.new(42)

# Determines Output Columns
men = Person.generate(4, "Man", r)
women = Person.generate(4, "Woman", r)

# Assume no Name is longer than 20 characters
spacer = '-' * (20 * 2 + 2)

# Solve the Problem
1.step do |round|
    singles = men.select(&:lonely?)
    singles.each do |m|
        m.propose(women)
    end

    break if singles.empty?

    puts "Round #{round}"
    puts spacer
    puts men, women
    puts spacer
end 
```

```
using Random

const mnames = ["A", "B", "C", "D"]
const wnames = ["E", "F", "G", "H"]

const Preferences = Dict{String,Vector{String}}
const Pairs = Dict{String,String}

# Returns a name => preference list dictionary, in decreasing order of preference
function genpreferences(mannames::Vector{String}, womannames::Vector{String})
    men   = Dict(map(m -> (m, shuffle(womannames)), mannames))
    women = Dict(map(w -> (w, shuffle(mannames)), womannames))
    return men, women
end

# Returns if `person` prefers the `first` candidate over the `second` one.
# This translates to `first` appearing *sooner* in the preference list
prefers(prefs, person, first, second) =
    findfirst(m -> m == first, prefs[person]) <
    findfirst(m -> m == second, prefs[person])

isfree(person, pairs) = !haskey(pairs, person)

function galeshapley(men::Preferences, women::Preferences)
    mentowomen = Dict{String,String}()
    womentomen = Dict{String,String}()
    while true
        bachelors = [m for m in keys(men) if isfree(m, mentowomen)]
        if length(bachelors) == 0
            return mentowomen, womentomen
        end

        for bachelor in bachelors
            for candidate in men[bachelor]
                if isfree(candidate, womentomen)
                    mentowomen[bachelor] = candidate
                    womentomen[candidate] = bachelor
                    break
                elseif prefers(women, candidate, bachelor, womentomen[candidate])
                    delete!(mentowomen, womentomen[candidate])
                    mentowomen[bachelor] = candidate
                    womentomen[candidate] = bachelor
                    break
                end
            end
        end
    end
end

function isstable(men::Preferences, women::Preferences, mentowomen::Pairs, womentoman::Pairs)
    for (husband, wife) in mentowomen
        for candidate in men[husband]
            if candidate != wife &&
               prefers(men, husband, candidate, wife) &&
               prefers(women, candidate, husband, womentoman[candidate])
                return false
            end
        end
    end
    return true
end

function main()
    men, women = genpreferences(mnames, wnames)
    mentowomen, womentomen = galeshapley(men, women)
    println(mentowomen)
    println(isstable(men, women, mentowomen, womentomen) ? "Stable" : "Unstable")
end

main() 
```

```
# Submitted by Marius Becker
# Updated by Amaras

from random import shuffle
from copy import copy
from string import ascii_uppercase, ascii_lowercase

def main():
    # Set this to however many men and women you want, up to 26
    num_pairs = 5

    # Create all Person objects
    men = [Person(name) for name in ascii_uppercase[:num_pairs]]
    women = [Person(name) for name in ascii_lowercase[:num_pairs]]

    # Set everyone's preferences
    for man in men:
        man.preference = copy(women)
        shuffle(man.preference)

    for woman in women:
        woman.preference = copy(men)
        shuffle(woman.preference)

    # Run the algorithm
    stable_marriage(men, women)

    # Print preferences and the result
    print('Preferences of the men:')
    for man in men:
        print(man)

    print()

    print('Preferences of the women:')
    for woman in women:
        print(woman)

    print('\n')

    print('The algorithm gave this solution:')
    for man in men:
        print(f'{man.name} + {man.partner.name}')

def stable_marriage(men, women):
    """Finds pairs with stable marriages"""

    while True:
        # Let every man without a partner propose to a woman
        for man in men:
            if not man.has_partner:
                man.propose_to_next()

        # Let the women pick their favorites
        for woman in women:
            woman.pick_preferred()

        # Continue only when someone is still left without a partner
        if all((man.has_partner for man in men)):
            return

class Person:

    def __init__(self, name):
        self.name = name
        self.preference = []
        self.candidates = []
        self.pref_index = 0
        self._partner = None

    @property
    def next_choice(self):
        """Return the next person in the own preference list"""
        try:
            return self.preference[self.pref_index]
        except IndexError:
            return None

    def propose_to_next(self):
        """Propose to the next person in the own preference list"""
        person = self.next_choice
        person.candidates.append(self)
        self.pref_index += 1

    def pick_preferred(self):
        """Pick a new partner or stay with the old one if they are preferred"""
        # Iterate own preferences in order
        for person in self.preference:
            # Pick the first person that's either a new candidate or the
            # current partner
            if person == self.partner:
                break
            elif person in self.candidates:
                self.partner = person
                break

        # Rejected candidates don't get a second chance
        self.candidates.clear()

    @property
    def partner(self):
        return self._partner

    # The call self.partner = person sets self._partner as person
    # However, since engagement is symmetrical, self._partner._partner
    # (which is then person._partner) also needs to be set to self
    @partner.setter
    def partner(self, person):
        """Set a person as the new partner and sets the partner of that
        person as well"""

        # Do nothing if nothing would change
        if person != self._partner:
            # Remove self from current partner
            if self._partner is not None:
                self._partner._partner = None

            # Set own and the other person's partner
            self._partner = person
            if self._partner is not None:
                self._partner._partner = self

    # This allows use of self.has_partner instead of self.has_partner()
    @property
    def has_partner(self):
        """Determine whether this person currently has a partner or not."""
        return self.partner is not None

    # This allows the preferences to be printed more elegantly
    def __str__(self):
        return f'{self.name}: {", ".join(p.name for p in self.preference)}'

if __name__ == '__main__':
    main() 
```

```
import           Data.Map as M (Map, (!))
import qualified Data.Map as M
import           Data.List (elemIndex)
import           Control.Monad.State

stableMatching :: (Ord a, Ord b) => [(a, [b])] -> [(b, [a])] -> [(a, b)]
stableMatching men women = evalState (propose (M.fromList women) men) M.empty

propose :: (Ord a, Ord b) => Map b [a] ->
                            [(a, [b])] ->
                            State (Map b (a, [b])) [(a, b)]
propose _ [] = get >>=  return . map (\(w, (m,_)) -> (m, w)) . M.assocs
propose women ((man, pref):bachelors) = do
  let theOne = head pref
  couples <- get
  case M.lookup theOne couples of
    Nothing -> do
      modify $ M.insert theOne (man, (tail pref))
      propose women bachelors
    Just (boyfriend, planB) -> do
      let rank x = elemIndex x (women!theOne)
      if rank boyfriend < rank man
        then propose women $ (man, tail pref): bachelors
        else do
          modify $ M.insert theOne (man, (tail pref)) . M.delete theOne
          propose women $ (boyfriend, planB): bachelors

main = do
  let aPref = [('A',"YXZ"), ('B',"ZYX"),('C', "XZY")]
      bPref = [('X',"BAC"), ('Y',"CBA"),('Z', "ACB")]
  print $ stableMatching aPref bPref 
```

```
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct person {
    size_t id;
    struct person *partner;
    size_t *prefers;
    size_t index;
};

void shuffle(size_t *array, size_t size) {
    for (size_t i = size - 1; i > 0; --i) {
        size_t j = (size_t)rand() % (i + 1);
        size_t tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
    }
}

void create_group(struct person *group, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        group[i].id = i;
        group[i].partner = NULL;
        group[i].prefers = malloc(sizeof(size_t) * size);
        group[i].index = 0;

        for (size_t j = 0; j < size; ++j) {
            group[i].prefers[j] = j;
        }

        shuffle(group[i].prefers, size);
    }
}

bool prefers_partner(size_t *prefers, size_t partner, size_t id, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (prefers[i] == partner) {
            return true;
        } else if(prefers[i] == id) {
            return false;
        }
    }
    return true;
}

void stable_marriage(struct person *men, struct person *women, size_t size) {
    struct person *bachelors[size];
    size_t bachelors_size = size;

    for (size_t i = 0; i < size; ++i) {
        bachelors[i] = &men[i];
    }

    while (bachelors_size > 0) {
        struct person *man = bachelors[bachelors_size - 1];
        struct person *woman = &women[man->prefers[man->index]];

        if (!woman->partner) {
            woman->partner = man;
            man->partner = woman;
            bachelors[--bachelors_size] = NULL;
        } else if (!prefers_partner(woman->prefers, woman->partner->id, man->id,
                                   size)) {

            woman->partner->index++;
            bachelors[bachelors_size - 1] = woman->partner;
            woman->partner = man;
            man->partner = woman;
        } else {
            man->index++;
        }
    }
}

void free_group(struct person *group, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        free(group[i].prefers);
    }
}

int main() {
    srand((unsigned int)time(NULL));

    struct person men[5], women[5];

    create_group(men, 5);
    create_group(women, 5);

    for (size_t i = 0; i < 5; ++i) {
        printf("preferences of man %zu: ", i);
        for (size_t j = 0; j < 5; ++j) {
            printf("%zu ", men[i].prefers[j]);
        }

        printf("\n");
    }

    printf("\n");

    for (size_t i = 0; i < 5; ++i) {
        printf("preferences of woman %zu: ", i);
        for (size_t j = 0; j < 5; ++j) {
            printf("%zu ", women[i].prefers[j]);
        }

        printf("\n");
    }

    stable_marriage(men, women, 5);

    printf("\n");

    for (size_t i = 0; i < 5; ++i) {
        printf("the partner of man %zu is woman %ld\n", i, men[i].partner->id);
    }

    free_group(men, 5);
    free_group(women, 5);
} 
```

```
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <vector>

// this header is so that we can use `not` and `and` on MSVC
#include <ciso646>

#include <cstddef>

using std::size_t;

/*
  we use these to generate random numbers in this program.
  this makes the program simpler,
  by not having to pass around random number generators.
*/
static thread_local std::random_device global_random_device;
static thread_local std::mt19937 global_rng(global_random_device());

struct person {
  /*
    this is a poor person's std::optional,
    but since we're attempting to be compileable on C++14,
    we won't worry too much about it.
  */
  bool finished;
  size_t preference;

  std::vector<size_t> preference_list;
};

/*
  this function generates a list of people with size `number_of_partners`.

  each person's `preference_list` will be a randomly sorted list of
  the numbers in the range [0, number_of_partners),
  with no duplicates.
*/
std::vector<person> make_person_list(size_t number_of_partners) {
  auto random_pref_list = [&] {
    std::vector<size_t> ret(number_of_partners);
    std::iota(begin(ret), end(ret), size_t(0));
    std::shuffle(begin(ret), end(ret), global_rng);

    return ret;
  };

  std::vector<person> ret;
  std::generate_n(std::back_inserter(ret), number_of_partners, [&] {
    return person{false, 0, random_pref_list()};
  });

  return ret;
}

template <typename LeadIter, typename FollowIter>
void stable_match(LeadIter leads, LeadIter leads_end, FollowIter follows) {
  // for each index in the leads' preference list, we'll go through this
  size_t const number_of_partners = leads_end - leads;
  for (size_t proposal_index = 0; proposal_index < number_of_partners;
       ++proposal_index) {
    /*
      each follow will get their own vector of proposals to them
      for each entry in the leads' proposal list

      if this weren't example code, this would likely go outside the loop
      to cut down on allocations
    */
    std::vector<std::vector<size_t>> proposals(number_of_partners);

    // for each lead, we'll make a proposal to their favorite follow
    for (size_t i = 0; i < number_of_partners; ++i) {
      if (not leads[i].finished) {
        auto pref = leads[i].preference_list[proposal_index];
        proposals[pref].push_back(i);
      }
    }

    // for each follow, we'll look at their preference list
    for (size_t i = 0; i < number_of_partners; ++i) {
      for (size_t pref : follows[i].preference_list) {
        for (size_t proposal : proposals[i]) {
          // and, if they were given a proposal, then they'll choose their
          // favorite here
          if (pref == proposal and not follows[i].finished) {
            follows[i].preference = pref;
            follows[i].finished = true;

            leads[pref].preference = i;
            leads[pref].finished = true;
          }
        }
      }
    }
  }
}

int main() {
  // these are the number of partners in each group
  size_t const number_of_partners = 5;

  // in this case, the leads shall propose to the follows
  auto leads = make_person_list(number_of_partners);
  auto follows = make_person_list(number_of_partners);

  stable_match(begin(leads), end(leads), begin(follows));

  // the happy marriages are announced to the console here :)
  for (size_t i = 0; i < number_of_partners; ++i) {
    std::cout << "the partnership of lead " << i << " and follow "
              << leads[i].preference << " shall commence forthwith!\n";
  }
} 
```

```
class Person {
  constructor(name) {
    this.name = name;
  }

  get hasFiance() {
    return !!this.fiance;
  }

  prefers(other) {
    return this.preferences.indexOf(other) < this.preferences.indexOf(this.fiance);
  }

  engageTo(other) {
    if (other.hasFiance) {
      other.fiance.fiance = undefined;
    }

    this.fiance = other;
    other.fiance = this;
  }
}

function stableMarriage(guys, girls) {
  const bachelors = [...guys];
  while (bachelors.length > 0) {
    const guy = bachelors.shift();
    for (const girl of guy.preferences) {
      if (!girl.hasFiance) {
        guy.engageTo(girl);
        break;
      } else if (girl.prefers(guy)) {
        bachelors.push(girl.fiance);
        guy.engageTo(girl);
        break;
      }
    }
  }
}

function shuffle(iterable) {
  const array = [...iterable];
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

const guys = [..."ABCDE"].map(name => new Person(name));
const girls = [..."FGHIJ"].map(name => new Person(name));

console.log("Guys");
for (const guy of guys) {
  guy.preferences = shuffle(girls);
  console.log(`${guy.name}: ${guy.preferences.map(p => p.name).join()}`)
}

console.log("\nGirls");
for (const girl of girls) {
  girl.preferences = shuffle(guys);
  console.log(`${girl.name}: ${girl.preferences.map(p => p.name).join()}`)
}

stableMarriage(guys, girls);

console.log("\nPairings");
for (const guy of guys) {
  console.log(`${guy.name}: ${guy.fiance.name}`);
} 
```

##### GaleShapleyAlgorithm.cs

```
// submitted by Julian Schacher (jspp) with great help by gustorn and Marius Becker
using System.Collections.Generic;

namespace StableMarriageProblem
{
    public static class GaleShapleyAlgorithm<TFollow, TLead>
        where TFollow : Person<TFollow, TLead>
        where TLead : Person<TLead, TFollow>
    {
        public static void RunGaleShapleyAlgorithm(List<TFollow> follows, List<TLead> leads)
        {
            // All follows are lonely.
            var lonelyFollows = new List<TFollow>(follows);

            // Carry on until there are no lonely follows anymore.
            while (lonelyFollows.Count > 0)
            {
                // Let every lonely follow propose to their current top choice.
                foreach (var lonelyFollow in lonelyFollows)
                {
                    lonelyFollow.ProposeToNext();
                }

                // Look which follows have a partner now and which don't.
                var newLonelyFollows = new List<TFollow>();
                foreach (var follow in follows)
                {
                    if (follow.Partner == null)
                        newLonelyFollows.Add(follow);
                }
                lonelyFollows = newLonelyFollows;
            }
        }
    }
} 
```

##### Person.cs

```
// submitted by Julian Schacher (jspp) with great help by gustorn and Marius Becker
using System.Collections.Generic;

namespace StableMarriageProblem
{
    public class Person<TSelf, TPref>
        where TSelf : Person<TSelf, TPref>
        where TPref : Person<TPref, TSelf>
    {
        public string Name { get; set; }
        public TPref Partner { get; set; }
        public IList<TPref> Choices { get; set; }
        // CurrentTopChoice equals the Choice in Choices that is the TopChoice,
        // if already tried women are not counted.
        public int CurrentTopChoiceIndex { get; set; } = 0;

        public Person(string name) => Name = name;

        public void ProposeToNext()
        {
            var interest = GetNextTopChoice();

            // If the interest has no partner or prefers this person,
            // change interest's partner to this person.
            if (interest.Partner == null ||
                interest.Choices.IndexOf(this as TSelf) < interest.Choices.IndexOf(interest.Partner))
            {
                // Should the interest already have a partner, set the partner
                // of the interest's partner to null.
                if (interest.Partner != null)
                    interest.Partner.Partner = null;
                interest.Partner = this as TSelf;
                Partner = interest;
            }
        }

        private TPref GetNextTopChoice() => Choices[CurrentTopChoiceIndex++];
    }
} 
```

##### Program.cs

```
// submitted by Julian Schacher (jspp) with great help by gustorn and Marius Becker
using System;
using System.Collections.Generic;

namespace StableMarriageProblem
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("GaleShapleyAlgorithm");
            // Using men and women as an example.
            var men = new List<Man>()
            {
                new Man("A"),
                new Man("B"),
                new Man("C"),
                new Man("D"),
                new Man("E")
            };
            var women = new List<Woman>()
            {
                new Woman("F"),
                new Woman("G"),
                new Woman("H"),
                new Woman("I"),
                new Woman("J"),
            };

            var random = new Random();

            foreach (var man in men)
            {
                man.Choices = new List<Woman>(women).Shuffle(random);
                Console.WriteLine(man.Name + ":");
                foreach (var choice in man.Choices)
                    Console.Write(choice.Name);
                Console.WriteLine();
            }
            foreach (var woman in women)
            {
                woman.Choices = new List<Man>(men).Shuffle(random);
                Console.WriteLine(woman.Name + ":");
                foreach (var choice in woman.Choices)
                    Console.Write(choice.Name);
                Console.WriteLine();
            }

            GaleShapleyAlgorithm<Woman, Man>.RunGaleShapleyAlgorithm(women, men);

            foreach (var woman in women)
            {
                Console.WriteLine(woman.Name + " : " + woman?.Partner.Name);
            }
        }
    }

    public class Man : Person<Man, Woman>
    {
        public Man(string name) : base(name) { }
    }

    public class Woman : Person<Woman, Man>
    {
        public Woman(string name) : base(name) { }
    }
} 
```

##### ListExtensions.cs

```
using System;
using System.Collections.Generic;

namespace StableMarriageProblem
{
    public static class ListExtensions
    {
        public static IList<T> Shuffle<T>(this IList<T> list, Random rng)
        {
            for (var i = 0; i < list.Count; i++)
            {
                var j = rng.Next(i, list.Count);
                var tmp = list[i];
                list[i] = list[j];
                list[j] = tmp;
            }
            return list;
        }
    }
} 
```

```
import java.util.List;
import java.util.ArrayList;
import java.util.Collections;

class StableMarriage {

    /*
     * Use the stable marriage algorithm to find stable pairs from the
     * lists of men and women.
     */
    public static void findStableMarriages(List<Woman> women, List<Man> men) {
        // We might have more men/women than women/men. In this case, not everybody can
        // get a mate. We should aim to give every member of the less numerous gender a mate,
        // as this is always possible.
        List<? extends Person> leastCommonGender = women.size() <= men.size() ? women : men;
        do {
            // Every single man proposes to a woman.
            for (Man man : men)
                if (man.isLonely())
                    man.propose();

            // The women pick their favorite suitor.
            for (Woman woman : women)
                woman.chooseMate();

            // End the process if everybody has a mate.
            if (!leastCommonGender.stream().anyMatch(Person::isLonely))
                break;

        } while (true);

        women.forEach(w -> System.out.println(w + " married to " + w.getMate()));
    }

    public static void main(String[] args) {
        int nPairs = 5;
        List<Woman> women = new ArrayList<>();
        List<Man> men = new ArrayList<>();
        for (char i = 'A'; i < 'A' + nPairs; ++i) {
            women.add(new Woman("" + i));
            men.add(new Man("" + i));
        }
        // Make the genders unbalanced:
        women.add(new Woman("X"));

        women.forEach(w -> {
            w.receiveOptions(men);
            System.out.println(w + " prefers " + w.getPreferredMates());
        });
        men.forEach(m -> {
            m.receiveOptions(women);
            System.out.println(m + " prefers " + m.getPreferredMates());
        });

        findStableMarriages(women, men);
    }

}

class Person {
    private final String name;
    protected Person mate;
    protected List<Person> preferredMates;

    public Person(String name) {
        this.name = name;
    }

    public boolean isLonely() {
        return mate == null;
    }

    public void setMate(Person mate) {
        // Only set mates if there is a change.
        if (this.mate != mate) {
            // Remove old mates mate.
            if (this.mate != null)
                this.mate.mate = null;

            // Set the new mate.
            this.mate = mate;

            // If new mate is someone, update their mate.
            if (mate != null)
                mate.mate = this;
        }
    }

    public Person getMate() {
        return mate;
    }

    public void receiveOptions(List<? extends Person> mates) {
        // Preferences are subjective.
        preferredMates = new ArrayList<>(mates);
        Collections.shuffle(preferredMates);
    }

    public List<Person> getPreferredMates() {
        return preferredMates;
    }

    public String toString() {
        return getClass().getName() + "(" + name + ")";
    }
}

class Woman extends Person {
    private List<Man> suitors = new ArrayList<>();

    public Woman(String name) {
        super(name);
    }

    public void recieveProposal(Man suitor) {
        suitors.add(suitor);
    }

    public void chooseMate() {
        for (Person mostDesired : preferredMates) {
            if (mostDesired == mate || suitors.contains(mostDesired)) {
                setMate(mostDesired);
                break;
            }
        }
    }
}

class Man extends Person {
    public Man(String name) {
        super(name);
    }

    public void propose() {
        if (!preferredMates.isEmpty()) {
            Woman fiance = (Woman) preferredMates.remove(0);
            fiance.recieveProposal(this);
        }
    }
} 
```

```
<?php
declare(strict_types=1);

class Person
{
    private $name;
    private $suitors = [];
    private $preferences = [];
    private $match;

    public function __construct($name)
    {
        $this->name = $name;
    }

    public function getName(): string
    {
        return $this->name;
    }

    public function setPreferences(array $preferences): void
    {
        $this->preferences = $preferences;
    }

    public function getMatch(): ?Person
    {
        return $this->match;
    }

    public function getPreferences(): array
    {
        return $this->preferences;
    }

    public function isSingle(): bool
    {
        return $this->match === null;
    }

    public function unmatch(): void
    {
        $this->match = null;
    }

    public function setMatch(Person $match): void
    {
        if ($this->match !== $match) {
            if ($this->match !== null) {
                $this->match->unmatch();
            }
            $this->match = $match;
            $match->setMatch($this);
        }
    }

    public function propose(): void
    {
        if (!empty($this->preferences)) {
            $fiance = array_shift($this->preferences);
            $fiance->receiveProposal($this);
        }
    }

    public function receiveProposal(Person $man): void
    {
        $this->suitors[] = $man;
    }

    public function chooseMatch(): void
    {
        foreach ($this->preferences as $preference) {
            if ($preference === $this->match || in_array($preference, $this->suitors)) {
                $this->setMatch($preference);
                break;
            }
        }

        $this->suitors = [];
    }

    public function __toString(): string
    {
        return $this->name;
    }
}

function stable_marriage(array $men, array $women): void
{
    do {
        foreach ($men as $man) {
            if ($man->isSingle()) {
                $man->propose();
            }
        }

        foreach ($women as $woman) {
            $woman->chooseMatch();
        }

        $unmarried = false;
        foreach ($women as $woman) {
            if ($woman->isSingle()) {
                $unmarried = true;
                break;
            }
        }

    } while ($unmarried);
}

$groupSize = 10;
$men = [];
$women = [];

for ($i = 1; $i <= $groupSize; $i++) {
    $men[] = new Person("M${i}");
    $women[] = new Person("W${i}");
}

foreach ($men as $man) {
    $preferences = $women;
    shuffle($preferences);
    $man->setPreferences($preferences);
    printf('%s\'s choices: %s', $man->getName(), implode(',', $man->getPreferences()));
    echo PHP_EOL;
}
echo PHP_EOL;
foreach ($women as $woman) {
    $preferences = $men;
    shuffle($preferences);
    $woman->setPreferences($preferences);
    printf('%s\'s choices: %s', $woman->getName(), implode(',', $woman->getPreferences()));
    echo PHP_EOL;
}
echo PHP_EOL;

stable_marriage($men, $women);
foreach ($women as $woman) {
    printf('%s is married to %s', $woman, $woman->getMatch());
    echo PHP_EOL;
} 
```

```
import scala.collection.mutable._

object StableMarriage {

  var bachelors = Queue[Man]()

  case class Man(name: String, var preferences: List[Woman] = List()) {
    def propose(): Unit = preferences match {
      case woman :: remainingPreferences => {
        if (woman.prefers(this)) {
          bachelors ++= woman.fiance
          woman.fiance = Some(this)
        }
        else
          bachelors.enqueue(this)
        preferences = remainingPreferences
      }
      case _ =>
    }
  }

  case class Woman(name: String, var preferences: List[Man] = List(), var fiance: Option[Man] = None) {
    def prefers(man: Man): Boolean =
      fiance match {
        case Some(otherMan) => preferences.indexOf(man) < preferences.indexOf(otherMan)
        case _ => true //always prefer any man over nobody
      }
  }

  def findStableMatches(men: Man*): Unit = {
    bachelors = men.to[Queue]
    while (bachelors.nonEmpty)
      bachelors.dequeue.propose()
  }
}

object StableMarriageExample {

  val a = StableMarriage.Man("Adam")
  val b = StableMarriage.Man("Bart")
  val c = StableMarriage.Man("Colm")
  val x = StableMarriage.Woman("Xena")
  val y = StableMarriage.Woman("Yeva")
  val z = StableMarriage.Woman("Zara")

  a.preferences = List(y, x, z)
  b.preferences = List(y, z, x)
  c.preferences = List(x, z, y)
  x.preferences = List(b, a, c)
  y.preferences = List(c, a, b)
  z.preferences = List(a, c, b)

  def main(args: Array[String]): Unit = {

    StableMarriage.findStableMatches(a, b, c)

    List(x, y, z).foreach(
      w => Console.println(
        w.name
          + " is married to "
          + w.fiance.getOrElse(StableMarriage.Man("Nobody")).name))
  }

} 
```

## 许可证

##### 代码示例

代码示例许可在 MIT 许可下（见[LICENSE.md](https://github.com/algorithm-archivists/algorithm-archive/blob/main/LICENSE.md)）。

##### 文本

本章的文本由[James Schloss](https://github.com/leios)编写，并许可在[Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/legalcode)下使用。

[]

![](https://creativecommons.org/licenses/by-sa/4.0/)

##### 提交的请求

在初始许可([#560](https://github.com/algorithm-archivists/algorithm-archive/pull/560))之后，以下提交的请求已修改本章的文本或图形：

+   无
