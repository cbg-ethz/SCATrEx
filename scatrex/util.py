import numpy
import numpy.random
import scipy.special
import scipy.stats
from functools import partial
import numpy as np

from jax.api import vmap
from jax import random
import jax.numpy as jnp
import jax.nn as jnn
from jax.scipy.stats import norm, gamma, laplace, beta, dirichlet, poisson
from jax.scipy.special import digamma, betaln

from pybiomart import Server
import pandas as pd


def relative_difference(current, prev, eps=1e-6):
    return (jnp.abs(current - prev) + eps) / (jnp.abs(prev) + eps)


def absolute_difference(current, prev):
    return jnp.abs(current - prev)


def diag_gamma_sample(rng, log_alpha, log_beta):
    return jnp.exp(-log_beta) * random.gamma(rng, jnp.exp(log_alpha))


def diag_gamma_logpdf(x, log_alpha, log_beta):
    # part = partial(gamma.logpdf, scale=jnp.exp(-log_beta))
    return jnp.sum(
        vmap(gamma.logpdf)(x=x, a=jnp.exp(log_alpha), scale=jnp.exp(-log_beta))
    )


def dirichlet_sample(rng, log_alpha):
    return random.dirichlet(rng, jnp.exp(log_alpha))


def dirichlet_logpdf(x, log_alpha):
    return dirichlet.logpdf(x, jnp.exp(log_alpha))


def diag_gaussian_sample(rng, mean, log_std):
    # Take a single sample from a diagonal multivariate Gaussian.
    return mean + jnp.exp(log_std) * random.normal(rng, mean.shape)


def diag_gaussian_logpdf(x, mean, log_std, axis=None):
    # Evaluate a single point on a diagonal multivariate Gaussian.
    return jnp.sum(vmap(norm.logpdf)(x, mean, jnp.exp(log_std)), axis=axis)


def loggaussian_logpdf(x, mean, std, axis=None):
    return (
        -jnp.log(x)
        - jnp.log(std)
        - 0.5 * jnp.log(2 * jnp.pi)
        - ((jnp.log(x) - mean) ** 2) / (2 * std**2)
    )


def diag_loggaussian_logpdf(x, mean, log_std, axis=None):
    # Evaluate a single point on a diagonal multivariate LogGaussian.
    return jnp.sum(vmap(loggaussian_logpdf)(x, mean, jnp.exp(log_std)), axis=axis)


def diag_laplace_sample(rng, mean, log_std):
    # Evaluate a single point on a diagonal multivariate Laplace.
    return mean + jnp.exp(log_std) * random.laplace(rng, mean.shape)


def diag_laplace_logpdf(x, mean, log_std, axis=None):
    # Evaluate a single point on a diagonal multivariate Laplace.
    return jnp.sum(vmap(laplace.logpdf)(x, mean, jnp.exp(log_std)), axis=axis)


def beta_sample(rng, log_alpha, log_beta):
    return random.beta(rng, jnp.exp(log_alpha), jnp.exp(log_beta))


def beta_logpdf(x, log_alpha, log_beta):
    return jnp.sum(vmap(beta.logpdf)(x, jnp.exp(log_alpha), jnp.exp(log_beta)))


# def gamma_sample(rng, log_alpha, log_beta):
#     return jnp.exp(-log_beta) * random.gamma(rng, jnp.exp(log_alpha))


def gamma_logpdf(x, alpha, beta):
    return jnp.sum(gamma.logpdf(x, alpha, scale=1.0 / beta))


#
# def gamma_logpdf(x, log_alpha):
#     return jnp.sum(vmap(gamma.logpdf)(x, jnp.exp(log_alpha)))


def digamma_eval(x):
    return vmap(digamma)(x)


def betaln_eval(x):
    return vmap(betaln)(x)


def convert_nb_params(mu, theta):
    """
    Convert mean/dispersion parameterization of a negative binomial to the ones scipy supports

    See https://en.wikipedia.org/wiki/Negative_binomial_distribution#Alternative_formulations
    """
    r = theta
    var = mu + 1 / r * mu**2
    p = (var - mu) / var
    return r, 1 - p


def negative_binomial_pmf(counts, mu, theta):
    """
    >>> import numpy as np
    >>> from scipy.stats import poisson
    >>> np.isclose(negative_binomial_pmf(10, 10, 10000), poisson.pmf(10, 10), atol=1e-3)
    True
    """
    return nbinom.pmf(counts, *convert_nb_params(mu, theta))


#
# def negative_binomial_pmf(counts, mu, theta):
#     return numpy.exp(negative_binomial_lpmf(counts, mu, theta))
#
# def negative_binomial_lpmf(counts, mu, theta):
#     res = gammaln(counts + theta) - gammaln(counts + 1) - gammaln(theta)
#     res = res + theta*numpy.log(theta) + counts*numpy.log(mu)
#     res = res + (theta + counts) * (-numpy.log(theta + mu))
#     return res


def negative_binomial_lpmf(counts, mu, theta):
    return jnp.sum(nbinom.logpmf(counts, *convert_nb_params(mu, theta)))


def negative_binomial_sample(mu, theta, size=None):
    return numpy.random.negative_binomial(*convert_nb_params(mu, theta), size=size)


def is_diag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)


def cmp(a, b):
    return (a > b) - (a < b)


def object2label(obj, nodes):
    labels = []
    for idx, item in enumerate(obj):
        for ii, nn in enumerate(nodes):
            if item == nn:
                labels.append(ii)
    labels = numpy.array(labels)
    return labels


def cluster_with_maxpear(X):
    try:
        import rpy2
    except:
        raise Exception(
            """Clustering with maximum PEAR requires rpy2 package. See http://rpy.sourceforge.net/rpy2.html."""
        )

    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri
    from rpy2.robjects.packages import importr

    rpy2.robjects.numpy2ri.activate()

    importr("mcclust")

    r = robjects.r

    robjects.globalenv["X"] = r.matrix(X, nrow=X.shape[0])

    clusters = r(
        """
          class(X) <- 'interger'
          clusters <- maxpear( comp.psm(X+1) )$cl;
       """
    )

    return clusters


def log_sum_exp(log_X):
    """
    Given a list of values in log space, log_X. Compute exp(log_X[0] + log_X[1] + ... log_X[n])

    Numerically safer than naive method.
    """
    max_exp = max(log_X)

    if numpy.isinf(max_exp):
        return max_exp

    total = 0

    for x in log_X:
        total += numpy.exp(x - max_exp)

    return numpy.log(total) + max_exp


def log_sum_exp_prod(log_X, Y):

    max_exp = max(log_X)

    if numpy.isinf(max_exp):
        return max_exp

    total = 0

    for x, y in zip(log_X, Y):
        total += numpy.exp(x - max_exp) * y

    return numpy.log(total) + max_exp


def sigmoid(x):
    res = 1.0 / (1.0 + numpy.exp(-x))
    return res


def logit(x):
    if x == 1.0:
        x = 1.0 - 1e-10
    if x == 0.0:
        x = 0.0 + 1e-10
    res = x / (1.0 - x)
    return res


def softmax(x):
    # Shift highest value to 0
    x = x - numpy.max(x)
    res = scipy.special.softmax(x)
    return res


def bucket(edges, value):
    return numpy.sum(value > edges)


def sticks_to_edges(sticks):
    return 1.0 - numpy.cumprod(1.0 - sticks)


def beta_mean_oflog(log_alpha, log_beta):
    alpha = jnp.exp(log_alpha)
    beta = jnp.exp(log_beta)
    return alpha / (alpha + beta)


def multinomial_sample(n, pvals, size=None):
    s = numpy.random.multinomial(n, pvals, size=size)
    return s


def multinomial_lpmf(x, n, pvals):
    return jnp.sum(scipy.stats.multinomial.logpmf(x.astype(int), n, pvals))


def poisson_sample(loc, size=None):
    s = numpy.random.poisson(loc, size=size)
    return s


def poisson_lpmf(x, loc, axis=None):
    return jnp.sum(poisson.logpmf(x.astype(int), loc), axis=axis)


def normal_sample(loc, scale, size=None):
    s = numpy.random.normal(loc, scale, size=size)
    return s


def mvn_sample(loc, cov, size=None):
    s = numpy.random.multivariate_normal(loc, cov, size=size)
    return s


def mvn_lpdf(x, loc, cov):
    # if is_diag(cov):
    #     return normal_lpdf(x, loc, np.diag(cov))
    var = scipy.stats.multivariate_normal(mean=loc, cov=cov)
    return var.logpdf(x)


def normal_lpdf(x, m, std):
    prec = 1 / std**2
    return jnp.sum(
        -0.5 * jnp.log(2 * jnp.pi)
        + 0.5 * jnp.log(prec + 1e-30)
        - 0.5 * prec * (x - m) ** 2
    )


def t_sample(df, loc, scale, size=None):
    s = scipy.stats.t.rvs(df, loc, scale, size=size)
    return s


def t_lpdf(x, df, loc, scale):
    return jnp.sum(scipy.stats.t.logpdf(x, df, loc, scale))


def mvl_sample(loc, cov, size=None):
    s = scipy.stats.laplace.rvs(loc=loc, scale=jnp.diag(cov), size=size)
    return s


def mvl_lpdf(x, loc, cov):
    # if is_diag(cov):
    #     return laplace_lpdf(x, loc, numpy.diag(cov))
    var = scipy.stats.laplace(loc=loc, scale=jnp.diag(cov))
    return jnp.sum(var.logpdf(x))


def laplace_sample(loc, scale, size=None):
    s = scipy.stats.laplace.rvs(loc, scale, size=size)
    return s


def laplace_lpdf(x, loc, scale):
    return numpy.sum(scipy.stats.laplace.logpdf(x, loc, scale))


def bernoulli_sample(prob, size=None):
    s = scipy.stats.bernoulli.rvs(prob, size=size)
    return s


def gamma_sample(shape, rate, size=None):
    s = numpy.random.gamma(shape, 1 / rate, size=size)
    return s


def gammaln(x):
    # small  = numpy.nonzero(x < numpy.finfo(numpy.float64).eps)
    result = scipy.special.gammaln(x)
    # result[small] = -numpy.log(x[small])
    return result


def gammapdfln(x, a, b):
    return -gammaln(a) + a * jnp.log(b) + (a - 1.0) * jnp.log(x) - b * x


def gamma_lpdf(x, shape, rate):
    return jnp.sum(gammapdfln(x, shape, rate))


def exp_gammapdfln(y, a, b):
    return a * jnp.log(b) - gammaln(a) + a * y - b * jnp.exp(y)


def betapdfln(x, a, b):
    if not isinstance(x, Iterable):
        x = jnp.array([x])
    if any(x == 0.0) or any(x == 1.0):
        return float("-inf")
    if any(x < 0.0) or any(x > 1.0):
        print(x)
    return (
        gammaln(a + b)
        - gammaln(a)
        - gammaln(b)
        + (a - 1.0) * jnp.log(x)
        + (b - 1.0) * jnp.log(1.0 - x)
    )


def boundbeta(a, b):
    return (1.0 - numpy.finfo(numpy.float64).eps) * (
        numpy.random.beta(a, b) - 0.5
    ) + 0.5
    # return numpy.random.beta(a,b)


def lnbetafunc(a):
    return numpy.sum(gammaln(a)) - gammaln(numpy.sum(a))


#
# def dirichlet_sample(concentration, size=None):
#     return numpy.random.dirichlet(concentration, size=size)


def dirichletpdfln(p, a):
    p[p == 0.0] = 1e-200
    p[p == 1.0] = 1.0 - 1e-200
    return -lnbetafunc(a) + numpy.sum((a - 1) * numpy.log(p))


def logsumexp(X, axis=None):
    maxes = numpy.max(X, axis=axis)
    return numpy.log(numpy.sum(numpy.exp(X - maxes), axis=axis)) + maxes


def convert_tidy_to_matrix(tidy_df):
    # Takes a tidy dataframe specifying the CNVs of cells along genomic bins
    # and converts it to a cell by bin matrix
    cell_df = tidy_df.loc[tidy_df.single_cell_id == tidy_df.single_cell_id[0]]
    bins_df = cell_df.drop(columns=["copy_number", "single_cell_id"], inplace=False)
    tidy_df["bin_id"] = np.tile(bins_df.index, tidy_df.single_cell_id.unique().size)
    matrix = tidy_df[["copy_number", "single_cell_id", "bin_id"]].pivot_table(
        values="copy_number", index="single_cell_id", columns="bin_id"
    )

    return matrix, bins_df


def annotate_bins(bins_df):
    # Takes a dataframe of genomic regions and returns an ordered list of full genes in each region
    server = Server("www.ensembl.org", use_cache=False)
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets["hsapiens_gene_ensembl"]
    gene_coordinates = dataset.query(
        attributes=[
            "chromosome_name",
            "start_position",
            "end_position",
            "external_gene_name",
        ],
        filters={
            "chromosome_name": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                "X",
                "Y",
            ]
        },
        use_attr_names=True,
    )
    # Drop duplicate genes
    gene_coordinates.drop_duplicates(subset="external_gene_name", ignore_index=True)

    annotated_bins = bins_df.copy()
    annotated_bins["genes"] = [list() for _ in range(annotated_bins.shape[0])]

    bin_size = bins_df["end"][0] - bins_df["start"][0]
    for index, row in gene_coordinates.iterrows():
        gene = row["external_gene_name"]
        if pd.isna(gene):
            continue
        start_bin_in_chr = int(row["start_position"] / bin_size)
        stop_bin_in_chr = int(row["end_position"] / bin_size)
        chromosome = str(row["chromosome_name"])
        chr_start = np.where(bins_df["chr"] == chromosome)[0][0]
        start_bin = start_bin_in_chr + chr_start
        stop_bin = stop_bin_in_chr + chr_start

        if stop_bin < annotated_bins.shape[0]:
            if np.all(annotated_bins.iloc[start_bin:stop_bin].chr == chromosome):
                for bin in range(start_bin, stop_bin + 1):
                    annotated_bins.loc[bin, "genes"].append(gene)

    return annotated_bins


def annotate_matrix(matrix, annotated_bins):
    # Takes a dataframe of cells by bins and a dataframe with gene lists for each bin
    # and returns a dataframe of cells by genes
    df_list = []
    chrs = []
    for bin, row in annotated_bins.iterrows():
        genes = row["genes"]
        chr = row["chr"]
        if len(genes) > 0:
            df_list.append(
                pd.concat([matrix[bin]] * len(genes), axis=1, ignore_index=True).rename(
                    columns=dict(zip(range(len(genes)), genes))
                )
            )
            chrs.append([chr] * df_list[-1].shape[1])
    chrs = np.concatenate(chrs)
    df = pd.concat(df_list, axis=1)
    chrs = chrs[np.where(~df.columns.duplicated())[0]]
    df = df.loc[:, ~df.columns.duplicated()]
    return df, chrs


def convert_phylogeny_to_clonal_tree(threshold):
    # Converts a phylogenetic tree to a clonal tree by choosing the main clades
    # according to some threshold
    raise NotImplementedError
