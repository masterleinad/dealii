/* ========================================================================== */
/* === UMF_valid_symbolic =================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) Timothy A. Davis, CISE,                              */
/* Univ. of Florida.  All Rights Reserved.  See ../Doc/License for License.   */
/* web: http://www.cise.ufl.edu/research/sparse/umfpack                       */
/* -------------------------------------------------------------------------- */

#include "umf_internal.h"

/* Returns TRUE if the Symbolic object is valid, FALSE otherwise. */
/* The UMFPACK_report_symbolic routine does a more thorough check. */

GLOBAL Int UMF_valid_symbolic
(
    SymbolicType *Symbolic
)
{
    /* This routine does not check the contents of the individual arrays, so */
    /* it can miss some errors.  All it checks for is the presence of the */
    /* arrays, and the Symbolic "valid" entry. */

    if (Symbolic == nullptr)
    {
	return (FALSE) ;
    }

    if (Symbolic->valid != SYMBOLIC_VALID)
    {
	/* Symbolic does not point to a SymbolicType object */
	return (FALSE) ;
    }

    if ((Symbolic->Cperm_init == nullptr) || (Symbolic->Rperm_init == nullptr) ||
	(Symbolic->Front_npivcol == nullptr) || (Symbolic->Front_1strow == nullptr) ||
	(Symbolic->Front_leftmostdesc == nullptr) ||
	(Symbolic->Front_parent == nullptr) || (Symbolic->Chain_start == nullptr) ||
	(Symbolic->Chain_maxrows == nullptr) || (Symbolic->Chain_maxcols == nullptr) ||
	Symbolic->n_row <= 0 || Symbolic->n_col <= 0)
    {
	return (FALSE) ;
    }

    return (TRUE) ;
}
