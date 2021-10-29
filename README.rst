.. vim: set fileencoding=utf-8 :
.. Sun 14 Aug 2016 17:56:41 CEST

.. image:: https://img.shields.io/badge/docs-v2.0.15-orange.svg
   :target: https://www.idiap.ch/software/bob/docs/bob/bob.io.audio/v2.0.15/index.html
.. image:: https://gitlab.idiap.ch/bob/bob.io.audio/badges/v2.0.15/pipeline.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio/commits/v2.0.15
.. image:: https://gitlab.idiap.ch/bob/bob.io.audio/badges/v2.0.15/coverage.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio/commits/v2.0.15
.. image:: https://img.shields.io/badge/gitlab-project-0000c0.svg
   :target: https://gitlab.idiap.ch/bob/bob.io.audio


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

Complete Bob's `installation`_ instructions. Then, to install this package,
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
