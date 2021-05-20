/* ========================================================================== */
/* === UMF_valid_numeric ==================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* UMFPACK Copyright (c) Timothy A. Davis, CISE,                              */
/* Univ. of Florida.  All Rights Reserved.  See ../Doc/License for License.   */
/* web: http://www.cise.ufl.edu/research/sparse/umfpack                       */
/* -------------------------------------------------------------------------- */

/* Returns TRUE if the Numeric object is valid, FALSE otherwise. */
/* Does not check everything.  UMFPACK_report_numeric checks more. */

#include "umf_internal.h"

GLOBAL Int UMF_valid_numeric
(
    NumericType *Numeric
)
{
    /* This routine does not check the contents of the individual arrays, so */
    /* it can miss some errors.  All it checks for is the presence of the */
    /* arrays, and the Numeric "valid" entry. */

    if (Numeric == nullptr)
    {
	return (FALSE) ;
    }

    if (Numeric->valid != NUMERIC_VALID)
    {
	/* Numeric does not point to a NumericType object */
	return (FALSE) ;
    }

    if (Numeric->n_row <= 0 || Numeric->n_col <= 0 || (Numeric->D == nullptr) ||
	(Numeric->Rperm == nullptr) || (Numeric->Cperm == nullptr) ||
	(Numeric->Lpos == nullptr) || (Numeric->Upos == nullptr) ||
	(Numeric->Lilen == nullptr) || (Numeric->Uilen == nullptr) || (Numeric->Lip == nullptr) || (Numeric->Uip == nullptr) ||
	(Numeric->Memory == nullptr) || (Numeric->ulen > 0 && (Numeric->Upattern == nullptr)))
    {
	return (FALSE) ;
    }

    return (TRUE) ;
}
