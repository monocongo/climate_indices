# How to contribute

Thanks for being here, we need volunteer developers to help this project come to fruition.

## Coding conventions

Hopefully a clear, coherent style will emerge from a reading of the code. We optimize for readability:

* Indent using four spaces (no tabs)
* Whitespaces after list items and method parameters (`[1, 2, 3]`, not `[1,2,3]`), around operators (`x += 1`, not `x+=1`), and around hash arrows.
* Liberal use of whitespace and blank lines
* Comment as much as necessary, more is better than less, and comments are code \-- maintain the commentary as if it were being compiled
* Underscores instead of camelCase
* This is open source software. Consider the people who will read your code, and make it look nice for them. It's sort of like driving a car: Perhaps you love doing donuts when you're alone, but with passengers the goal is to make the ride as smooth as possible.

## Issues
The list of outstanding feature requests and bugs can be found on our on our GitHub issue tracker. Pick an unassigned issue that you think you can accomplish, add a comment that you are attempting to do it, and shortly your own personal label matching your GitHub ID will be assigned to that issue.

Feel free to propose issues that aren’t described.

## Setting up topic branches and generating pull requests
While it’s handy to provide useful code snippets in an issue, it is better for you as a developer to submit pull requests. By submitting pull request your contribution will be recorded by Github.

In git it is best to isolate each topic or feature into a “topic branch”. While individual commits allow you control over how small individual changes are made to the code, branches are a great way to group a set of commits all related to one feature together, or to isolate different efforts when you might be working on multiple topics at the same time.

While it takes some experience to get the right feel about how to break up commits, a topic branch should be limited in scope to a single issue as submitted to an issue tracker.

Also since GitHub pegs and syncs a pull request to a specific branch, it is the ONLY way that you can submit more than one fix at a time. If you submit a pull from your develop branch, you can’t make any more commits to your develop without those getting added to the pull.

To create a topic branch, its easiest to use the convenient -b argument to git checkout:

`git checkout -b fix-broken-thing`

Switched to a new branch 'fix-broken-thing'

You should use a verbose enough name for your branch so it is clear what it is about. Now you can commit your changes and regularly merge in the upstream develop as described below.

When you are ready to generate a pull request, either for preliminary review, or for consideration of merging into the project you must first push your local topic branch back up to GitHub:

`git push origin fix-broken-thing`

Now when you go to your fork on GitHub, you will see this branch listed under the “Source” tab where it says “Switch Branches”. Go ahead and select your topic branch from this list, and then click the “Pull request” button.

Here you can add a comment about your branch. If this in response to a submitted issue, it is good to put a link to that issue in this initial comment. The repository managers will be notified of your pull request and it will be reviewed (see below for best practices). Note that you can continue to add commits to your topic branch (and push them up to GitHub) either if you see something that needs changing, or in response to a reviewer’s comments. If a reviewer asks for changes, you do not need to close the pull and reissue it after making changes. Just make the changes locally, push them to GitHub, then add a comment to the discussion section of the pull request.

## Pull upstream changes into your fork regularly
It is critical that you pull upstream changes from develop branch into your fork on a regular basis. Nothing is worse than putting in a days of hard work into a pull request only to have it rejected because it has diverged too far from the main branch.

To pull in upstream changes:

`git remote add upstream https://github.com/monocongo/climate_indices.git`
`git fetch upstream develop`

Check the log to be sure that you actually want the changes, before merging:

`git log upstream/develop`

Then merge the changes that you fetched:

`git merge upstream/develop`

For more info, see http://help.github.com/fork-a-repo/

## How to get your pull request accepted
We want your submission. But we also want to provide a stable experience for our users and the community.
Follow these rules and you should succeed without a problem.

### Run the tests
Before you submit a pull request, please run the entire test suite via:

`$ export NUMBA_DISABLE_JIT=1`
`$ python setup.py test`
`$ unset  NUMBA_DISABLE_JIT`

The first thing the core committers will do is run this command. Any pull request that fails this test suite will be rejected.

### If you add code you need to add tests
We’ve learned the hard way that code without tests is undependable. If your pull request reduces our test coverage because it lacks tests then it will be rejected.

Also, keep your tests as simple as possible. Complex tests end up requiring their own tests. We would rather see duplicated assertions across test methods then cunning utility methods that magically determine which assertions are needed at a particular stage. Remember: Explicit is better than implicit.

### Clarity
Always write a clear log message for your commits. One-line messages are fine for small changes, but bigger changes should look like this:

`$ git commit -m "A brief summary of the commit
    \>
    \> A paragraph describing what changed and its impact.`

### Don’t mix code changes with whitespace cleanup
If you change two lines of code and correct 200 lines of whitespace issues in a file then the diff on that pull request is functionally unreadable and will be rejected. Whitespace cleanups need to be in their own pull request.

### Keep your pull requests limited to a single issue
Pull requests should be as small/atomic as possible. Large, wide-sweeping changes in a pull request will be rejected, with comments to isolate the specific code in your pull request.

### Follow PEP-8 and keep your code simple
Memorize the Zen of Python:

`>>> python -c 'import this'`

Please keep your code as clean and straightforward as possible. When we see more than one or two functions/methods starting with \_my_special_function or things like \__builtins\__.object = str we start to get worried. Rather than try and figure out your brilliant work we’ll just reject it and send along a request for simplification.

Furthermore, the pixel shortage is over. We want to see:

`package` instead of `pkg`
`grid` instead of `g`
`my_function_that_does_things` instead of `mftdt`

##How pull requests are checked, tested, and done
First we pull the code into a local branch:

`git checkout -b <branch-name> <submitter-github-name>`

`git pull git://github.com/<submitter-github-name/climate_indices.git develop`

Then we run the tests:

`$ python -m unittest tests/test_*.py`

We finish with a merge and push to GitHub:

`git checkout develop`
`git merge <branch-name>`
`git push origin develop`

# Thanks for your help and participation!
