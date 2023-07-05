.. raw:: html

    <p>
      <h1>
        <a href="https://checkers_hedm.readthedocs.io"><img valign="middle" src="https://raw.githubusercontent.com/argerlt/checkers/main/doc/_static/img/Checkers_Logo.png" width="50" alt="temporary checkers logo"/></a>
        Checkers
      </h1>
    </p>

.. Anything above the word EXCLUDE at the bottom of this comment will
.. be excluded from checker's long discription. Right now, that is meaningless, but
.. if this ever becomes a PyPI project, you will want to exclude the title banner.
.. Also, you can put fun comments here. Simon is a nice guy.

.. EXCLUDE

|build_status|_ |Coveralls|_ |docs|_ |black|_

.. |build_status| image:: https://github.com/pyxem/orix/workflows/build/badge.svg
.. _build_status: https://github.com/pyxem/orix/actions

.. |Coveralls| image:: https://coveralls.io/repos/github/pyxem/orix/badge.svg?branch=develop
.. _Coveralls: https://coveralls.io/github/pyxem/orix?branch=develop

.. |docs| image:: https://readthedocs.org/projects/orix/badge/?version=latest
.. _docs: https://orix.readthedocs.io/en/latest

.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
.. _black: https://github.com/psf/black


Checkers is a rudimentary python module for working with HEXRD data, specificially
at the `Cornell High Energy Synchrotron Source (CHESS) <https://www.chess.cornell.edu/>`_.

The module includes, amongst other tools, a Virtual Diffractometer for both near and 
far field data. The Tutorials are intended to help walk new users though the 
information necessary to understand and process their synchrotron data as well, while 
Scripts includes some experimental example workflows.

Due to it's dependencies, checkers is released under the GPL v3 license. It is also VERY
much in Alpha, so users beware, as functionality might break with little notice.


Motivation
----------

*"If CHESS seems hard, try learning CHECKERS first"*

Checker's purpose is three-fold:

1) Provide a well-documented open source Virtual Diffraction tool.
2) Use as a testbed for prototyping new 3DXRD analysis tools.
3) Sharing data processing pipelines between HEXRD users.

Notably, Checkers is NOT meant as a replacement for `HEXRD <https://github.com/HEXRD>`_, 
`Midas <https://www.aps.anl.gov/Science/Scientific-Software/MIDAS>`_, or other mature 
3DXRD analysis suites. Consider this more of a "beginners version", for new users not 
already intimately familiar with diffaction techniques, terminology, 
and algorithms. 

Installation
------------

checkers can be installed from the github source using ``git`` and ``pip`` as follows::
    
    git clone https://github.com/argerlt/checkers.git
    cd checkers
    pip install --editable .


Contributing
------------

TODO: add contribution guidelines

Documentation
-------------

TODO: add contribution guidelines


Motivation, continued
---------------------

It has been my experience that existing codebases (of which I am most familiar with HEXRD) are
difficult both to learn from or contribute to for new users, due to a lack of
well documented open source code. This is understandable, as many of these codes were rapidly
developed alongside the techniques themselves by large teams of contributers, all racing to stay on
the cutting edge.

However, as techniques become more standardized, it makes sense to take a more organized
approach to building a stable (if somewhat basic) codebase for 3DXRD analysis. A lot of 
inpsiration for how to implement this is taken from ORIX, including the implementation 
of unittests, coverall, python black, sphinx, github workflows, and readthedocs. Big thanks
to the pyxem team.


