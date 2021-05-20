/* ========================================================================= */
/* === AMD_order =========================================================== */
/* ========================================================================= */

/* ------------------------------------------------------------------------- */
/* AMD, Copyright (c) Timothy A. Davis,					     */
/* Patrick R. Amestoy, and Iain S. Duff.  See ../README.txt for License.     */
/* email: davis at cise.ufl.edu    CISE Department, Univ. of Florida.        */
/* web: http://www.cise.ufl.edu/research/sparse/amd                          */
/* ------------------------------------------------------------------------- */

/* User-callable AMD minimum degree ordering routine.  See amd.h for
 * documentation.
 */

#include "amd_internal.h"

/* ========================================================================= */
/* === AMD_order =========================================================== */
/* ========================================================================= */

GLOBAL Int AMD_order
(
    Int n,
    const Int Ap [ ],
    const Int Ai [ ],
    Int P [ ],
    double Control [ ],
    double Info [ ]
)
{
    Int *Len, *S, nz, i, *Pinv, info, status, *Rp, *Ri, *Cp, *Ci, ok ;
    size_t nzaat, slen ;
    double mem = 0 ;

#ifndef NDEBUG
    AMD_debug_init ("amd") ;
#endif

    /* clear the Info array, if it exists */
    info = static_cast<long>(Info != (double *) NULL) ;
    if (info != 0)
    {
	for (i = 0 ; i < AMD_INFO ; i++)
	{
	    Info [i] = EMPTY ;
	}
	Info [AMD_N] = n ;
	Info [AMD_STATUS] = AMD_OK ;
    }

    /* make sure inputs exist and n is >= 0 */
    if (Ai == (Int *) NULL || Ap == (Int *) NULL || P == (Int *) NULL || n < 0)
    {
	if (info != 0) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;	    /* arguments are invalid */
    }

    if (n == 0)
    {
	return (AMD_OK) ;	    /* n is 0 so there's nothing to do */
    }

    nz = Ap [n] ;
    if (info != 0)
    {
	Info [AMD_NZ] = nz ;
    }
    if (nz < 0)
    {
	if (info != 0) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;
    }

    /* check if n or nz will cause size_t overflow */
    if (((size_t) n) >= SIZE_T_MAX / sizeof (Int)
     || ((size_t) nz) >= SIZE_T_MAX / sizeof (Int))
    {
	if (info != 0) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;	    /* problem too large */
    }

    /* check the input matrix:	AMD_OK, AMD_INVALID, or AMD_OK_BUT_JUMBLED */
    status = AMD_valid (n, n, Ap, Ai) ;

    if (status == AMD_INVALID)
    {
	if (info != 0) Info [AMD_STATUS] = AMD_INVALID ;
	return (AMD_INVALID) ;	    /* matrix is invalid */
    }

    /* allocate two size-n integer workspaces */
    Len = (Int*)amd_malloc (n * sizeof (Int)) ;
    Pinv = (Int*)amd_malloc (n * sizeof (Int)) ;
    mem += n ;
    mem += n ;
    if ((Len == nullptr) || (Pinv == nullptr))
    {
	/* :: out of memory :: */
	amd_free (Len) ;
	amd_free (Pinv) ;
	if (info != 0) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;
    }

    if (status == AMD_OK_BUT_JUMBLED)
    {
	/* sort the input matrix and remove duplicate entries */
	AMD_DEBUG1 (("Matrix is jumbled\n")) ;
	Rp = (Int*)amd_malloc ((n+1) * sizeof (Int)) ;
	Ri = (Int*)amd_malloc (MAX (nz,1) * sizeof (Int)) ;
	mem += (n+1) ;
	mem += MAX (nz,1) ;
	if ((Rp == nullptr) || (Ri == nullptr))
	{
	    /* :: out of memory :: */
	    amd_free (Rp) ;
	    amd_free (Ri) ;
	    amd_free (Len) ;
	    amd_free (Pinv) ;
	    if (info != 0) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	    return (AMD_OUT_OF_MEMORY) ;
	}
	/* use Len and Pinv as workspace to create R = A' */
	AMD_preprocess (n, Ap, Ai, Rp, Ri, Len, Pinv) ;
	Cp = Rp ;
	Ci = Ri ;
    }
    else
    {
	/* order the input matrix as-is.  No need to compute R = A' first */
	Rp = NULL ;
	Ri = NULL ;
	Cp = (Int *) Ap ;
	Ci = (Int *) Ai ;
    }

    /* --------------------------------------------------------------------- */
    /* determine the symmetry and count off-diagonal nonzeros in A+A' */
    /* --------------------------------------------------------------------- */

    nzaat = AMD_aat (n, Cp, Ci, Len, P, Info) ;
    AMD_DEBUG1 (("nzaat: %g\n", (double) nzaat)) ;
    ASSERT ((MAX (nz-n, 0) <= nzaat) && (nzaat <= 2 * (size_t) nz)) ;

    /* --------------------------------------------------------------------- */
    /* allocate workspace for matrix, elbow room, and 6 size-n vectors */
    /* --------------------------------------------------------------------- */

    S = NULL ;
    slen = nzaat ;			/* space for matrix */
    ok = static_cast<long>((slen + nzaat/5) >= slen) ;	/* check for size_t overflow */
    slen += nzaat/5 ;			/* add elbow room */
    for (i = 0 ; (ok != 0) && i < 7 ; i++)
    {
	ok = static_cast<long>((slen + n) > slen) ;	/* check for size_t overflow */
	slen += n ;			/* size-n elbow room, 6 size-n work */
    }
    mem += slen ;
    ok = static_cast<int>((static_cast<long>(ok != 0) != 0) && (slen < SIZE_T_MAX / sizeof (Int))) ; /* check for overflow */
    ok = static_cast<int>((static_cast<long>(ok != 0) != 0) && (slen < Int_MAX)) ;	/* S[i] for Int i must be OK */
    if (ok != 0)
    {
	S = (Int*)amd_malloc (slen * sizeof (Int)) ;
    }
    AMD_DEBUG1 (("slen %g\n", (double) slen)) ;
    if (S == nullptr)
    {
	/* :: out of memory :: (or problem too large) */
	amd_free (Rp) ;
	amd_free (Ri) ;
	amd_free (Len) ;
	amd_free (Pinv) ;
	if (info != 0) Info [AMD_STATUS] = AMD_OUT_OF_MEMORY ;
	return (AMD_OUT_OF_MEMORY) ;
    }
    if (info != 0)
    {
	/* memory usage, in bytes. */
	Info [AMD_MEMORY] = mem * sizeof (Int) ;
    }

    /* --------------------------------------------------------------------- */
    /* order the matrix */
    /* --------------------------------------------------------------------- */

    AMD_1 (n, Cp, Ci, P, Pinv, Len, slen, S, Control, Info) ;

    /* --------------------------------------------------------------------- */
    /* free the workspace */
    /* --------------------------------------------------------------------- */

    amd_free (Rp) ;
    amd_free (Ri) ;
    amd_free (Len) ;
    amd_free (Pinv) ;
    amd_free (S) ;
    if (info != 0) Info [AMD_STATUS] = status ;
    return (status) ;	    /* successful ordering */
}
