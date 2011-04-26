========================================
 Golden Rules for |project| Development
========================================

Developing against |project| should be a joy! Nevertheless, all software
projects need some ground rules (or **Golden Rules**) to guarantee minimal
usability and maintainability. Here are ours:

1. The latest version of our version control repository (sometimes referred as
   HEAD) on the main development branch, must **always** compile w/o
   problems. This means you **cannot** check-in broken code. You must **at
   least** make sure it compiles on all supported platforms;
2. There must always be unit tests that cover new code. New code is not
   acceptable otherwise;
3. New code must always come with extensive documentation. We have put in place
   tools to assure that code that breaks these rules can be detected where
   possible.
