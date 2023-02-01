import numpy as np
import scipy


def column_selection_sp(mat_o, d, k, seed=31, MAX_ITER=1000, ZERO=1e-10):
    mat = np.asmatrix(mat_o, dtype=np.float32)
    M, N = mat.shape
    if M > N:
        _, s, Vh = scipy.linalg.svd(mat, lapack_driver='gesvd')
        S = np.diag(s)
        sigma_vt = S.dot(Vh)
        A = np.asmatrix(sigma_vt[:N, :])
    else:
        A = np.asmatrix(mat)

    k_list = []
    loss_list = []
    colIDs_list = []

    np.random.seed(seed=seed)
    converged = False
    previous_norm = 0
    iterations = 0
    current_subset = np.random.choice(N, k, False)

    C = np.asmatrix(A[:, current_subset])
    cpinv = np.asmatrix(scipy.linalg.pinv(C))
    E = A - C.dot(cpinv.dot(A))

    previous_subset = current_subset

    if N > M:
        # _, s_E, Vh_E = scipy.linalg.svd(E, lapack_driver='gesvd')
        #
        raise NotImplementedError
    else:
        F = np.dot(E.T, E)
        f = np.sum(np.power(F, 2), axis=0)
        g = np.diag(F).transpose()

    current_iter = 0
    while (not converged) and current_iter < MAX_ITER:
        for i in range(k):
            j = current_subset[i]

            # Update pseudoinverse
            Ct = A[:, current_subset]
            Ct[:, i] = 0
            cPinvAi = np.dot(cpinv[i, :], A)
            CpCpi = np.asmatrix(np.dot(cpinv, cpinv[i, :].T))
            nn = CpCpi[i]
            # print cpinv.shape, CpCpi.shape, nn
            cpinv = cpinv - np.multiply(1/nn, np.dot(CpCpi, cpinv[i, :]))

            # Update residual
            CtGA = np.multiply(-1/nn, np.dot(Ct, np.dot(CpCpi, cPinvAi)))
            pre_residual = E
            E = E + np.dot(A[:, j], cPinvAi) - CtGA
            # Update scores to account for missing one
            delta = np.dot(E[:, j].T, E)
            delta_norm = np.linalg.norm(delta, ord=2)
            delta_norm = delta_norm * delta_norm
            gamma = np.dot(pre_residual.T, np.dot(pre_residual, delta.T))
            delta_sq = np.power(delta, 2)
            deltaj_sq = delta[0,j]*delta[0,j]
            f = f + delta_norm*(delta_sq/deltaj_sq) + 2.0 * np.multiply(gamma.T, delta/delta[0,j])
            g = g + delta_sq/delta[0,j]
            scores = np.divide(f, g)

            # print E.shape, delta.shape, scores.shape
            # Set the scores of already present columns to 0
            for h in current_subset.T:
                if h != current_subset[i]:
                    scores[0, h] = 0

            # To avoid numerical issues
            g = np.where(g < ZERO, 0, g)

            # Choose the winning feature
            winner = np.argmax(scores)
            current_subset[i] = winner

            # Update scores to account for winner
            delta = np.dot(E[:, winner].T, E)
            delta_norm = np.linalg.norm(delta, ord=2)
            delta_norm = delta_norm * delta_norm
            gamma = np.dot(E.T, np.dot(E, delta.T))
            delta_sq = np.power(delta, 2)
            deltaj_sq = delta[0,winner] * delta[0,winner]
            f = f + delta_norm*(delta_sq/deltaj_sq) - 2.0 * np.multiply(gamma.T, delta/delta[0,winner])
            g = g - delta_sq/delta[0,winner]

            # Update pseudoinverse
            v = np.dot(cpinv, A[:, winner])
            w = E[:, winner]
            nw = np.dot(w.T, w)
            third = np.multiply(1/nw, w.T)
            G = np.multiply(-1/nw, np.dot(v, w.T))
            G[i, :] = third
            cpinv = cpinv + G
            E = E - np.divide(np.dot(E[:, winner], np.dot(E[:, winner].T, E)), np.dot(E[:, winner].T, E[:, winner]))

        if MAX_ITER > 1:
            C = A[:, current_subset]
            cnorm = np.linalg.norm(np.dot(C, np.dot(scipy.linalg.pinv(C), A)), ord="fro")
            if cnorm > previous_norm:
                previous_norm = cnorm
                previous_subset = current_subset
            else:
                converged = True
                current_subset = previous_subset

            iterations += 1

        current_iter += 1

    R = current_subset
    R.sort()
    C = mat_o[:, R]
    # loss = np.linalg.norm(mat - np.dot(C, np.dot(scipy.linalg.pinv(C), mat)), ord="fro")

    return C, R


