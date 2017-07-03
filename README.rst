.. vim: set fileencoding=utf-8 :
.. Sun 14 Aug 2016 17:56:41 CEST

.. image:: http://img.shields.io/badge/docs-stable-yellow.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.io.audio/stable/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.io.audio/master/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.io.audio/badges/master/build.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio/commits/master
.. image:: https://gitlab.idiap.ch/bob/bob.io.audio/badges/master/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio/commits/master
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio
.. image:: http://img.shields.io/pypi/v/bob.io.audio.svg
   :target: https://pypi.python.org/pypi/bob.io.audio
.. image:: http://img.shields.io/pypi/dm/bob.io.audio.svg
   :target: https://pypi.python.org/pypi/bob.io.audio


===========================
 Audio I/O Support for Bob
===========================

This package is part of the signal-processing and machine learning toolbox
Bob_. It contains support for Audio I/O in Bob. Audio reading and writing is
implemented using SoX. By importing this package, you activate a transparent
plugin that makes possible reading and writing audio files using
``bob.io.base`` functionalities.


Installation
------------

Complete Bob's `installation`_ instructions. Then, to install this pacakge,
run::

  $ conda install bob.io.audio


Contact
-------

For questions or reporting issues to this software package, contact our
development `mailing list`_.


.. Place your references here:
.. _bob: https://www.idiap.ch/software/bob
.. _installation: https://www.idiap.ch/software/bob/install
.. _mailing list: https://www.idiap.ch/software/bob/discuss
