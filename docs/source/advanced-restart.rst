Restarting Parameter Fits
=========================

The ``rsdfit`` executable includes a ``restart`` sub-command for restarting
parameter fits from existing parameter fit, which can be either a MCMC or NLOPT
``.npz`` result file. To restart from a specific result, simply pass the name
of the result file and specify the appropriate model to load and the
number of additional iterations to run.

The calling sequence is:

.. command-output:: rsdfit restart -h

The code will run the number of additional iterations specified by the user
via the ``-i`` flag. Upon finishing, a new results file holding the
concatenation of the result that was restarted and the new result will
be written to file. Then, lastly, the original result file that was restarted
will be deleted.

.. note::

    It is possible to specify multiple result files on the command line to
    restart. In this case, a parameter fit will be restarted for each file passed,
    and the fits will run in parallel. Thus, there must be enough parallel
    MPI processes available to run each restart file specified on the command
    line. If this isn't the case, the code will crash with an error.
