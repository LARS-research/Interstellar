import numpy as np

class CategoricalASNG:
    """
    code refers to https://github.com/shirakawas/ASNG-NAS
    """

    def __init__(self, categories, alpha=1.5, delta_init=1., lam=2., Delta_max=10, init_theta=None):
        if init_theta is not None:
            self.theta = init_theta


        self.N = np.sum(categories - 1)
        self.d = len(categories)
        self.C = categories
        self.Cmax = np.max(categories)
        self.theta = np.zeros((self.d, self.Cmax))

        for i in range(self.d):
            self.theta[i,:self.C[i]] = 1./self.C[i]

        for i in range(self.d):
            self.theta[i, self.C[i]:] = 0.

        self.valid_param_num = int(np.sum(self.C - 1))
        self.valid_d = len(self.C[self.C>1])


        self.alpha = alpha
        self.delta_init = delta_init
        self.lam = lam
        self.Delta_max = Delta_max

        self.Delta = 1.
        self.gamma = 0.0
        self.delta = self.delta_init / self.Delta
        self.eps = self.delta
        self.s = np.zeros(self.N)

    def get_lam(self):
        return self.lam

    def get_delta(self):
        return self.delta / self.Delta

    def sampling(self):
        rand = np.random.rand(self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)

        c = (cum_theta - self.theta<=rand) & (rand<cum_theta)
        return c

    def sampling_lam(self, lam):
        rand = np.random.rand(lam, self.d, 1)
        cum_theta = self.theta.cumsum(axis=1)

        c = (cum_theta - self.theta<=rand) & (rand<cum_theta)
        return c

    def mle(self):
        m = self.theta.argmax(axis=1)
        x = np.zeros((self.d, self.Cmax))
        for i,c in enumerate(m):
            x[i,c] = 1
        return x

    def update(self, c_one, fxc, range_restriction=True):
        delta = self.get_delta()
        #print('delta:', delta)
        beta = self.delta * self.N**(-0.5)

        aru, idx = self.utility(fxc)
        mu_W, var_W = aru.mean(), aru.var()
        #print(fxc, idx, aru)
        if var_W == 0:
            return

        ngrad = np.mean((aru-mu_W)[:, np.newaxis, np.newaxis] * (c_one[idx] - self.theta), axis=0)
    
        if (np.abs(ngrad) < 1e-18).all():
            #print('skip update')
            return

        sl = []
        for i, K in enumerate(self.C):
            theta_i = self.theta[i, :K-1]
            theta_K = self.theta[i, K-1]
            s_i = 1./np.sqrt(theta_i) * ngrad[i, :K-1]
            s_i += np.sqrt(theta_i) * ngrad[i, :K-1].sum() / (theta_K + np.sqrt(theta_K))
            sl += list(s_i)
        sl = np.array(sl)

        ngnorm = np.sqrt(np.dot(sl, sl)) + 1e-8
        dp = ngrad / ngnorm
        assert not np.isnan(dp).any(), (ngrad, ngnorm)

        self.theta += delta * dp

        self.s = (1-beta) * self.s + np.sqrt(beta * (2-beta)) * sl / ngnorm
        self.gamma = (1-beta)**2 * self.gamma + beta*(2-beta)
        self.Delta *= np.exp(beta * (self.gamma - np.dot(self.s, self.s) / self.alpha))
        self.Delta = min(self.Delta , self.Delta_max)

        for i in range(self.d):
            ci = self.C[i]
            theta_min = 1./(self.valid_d*(ci-1)) if range_restriction and ci>1 else 0.
            self.theta[i, :ci] = np.maximum(self.theta[i, :ci], theta_min)
            theta_sum = self.theta[i, :ci].sum()
            tmp = theta_sum - theta_min * ci
            self.theta[i, :ci] -= (theta_sum-1.) * (self.theta[i,:ci]-theta_min)/tmp
            self.theta[i, :ci] /= self.theta[i,:ci].sum()

    def get_arch(self,):
        return np.argmax(self.theta, axis=1)

    def get_max(self,):
        return np.max(self.theta, axis=1)

    def get_entropy(self,):
        ent = 0
        for i, K in enumerate(self.C):
            the = self.theta[i,:K]
            ent += np.sum(the*np.log(the))
        return -ent

    @staticmethod
    def utility(f, rho=0.25, negative=True):
        eps = 1e-3
        idx = np.argsort(f)
        lam = len(f)
        mu = int(np.ceil(lam * rho))
        _w = np.zeros(lam)
        _w[:mu] = 1/mu
        _w[lam-mu:] = -1/mu if negative else 0
        w = np.zeros(lam)
        istart = 0
        for i in range(len(f) - 1):
            if f[idx[i+1]] - f[idx[i]] < eps*f[idx[i]]:
                pass
            elif istart < i:
                w[istart:i+1] = np.mean(_w[istart:i+1])
                istart = i+1
            else:
                w[i] = _w[i]
                istart = i+1
        w[istart:] = np.mean(_w[istart:])
        return w, idx

    def log_header(self, theta_log=False):
        header_list = ['delta', 'eps', 'snorm_alha', 'theta_converge']
        if theta_log:
            for i in range(self.d):
                header_list += ['theta%d_%d' % (i,j) for j in range(self.C[i])]
        return header_list

    def log(self, theta_log=False):
        log_list = [self.delta, self.eps, np.dot(self.s, self.s)/self.alpha, self.theta.max(axis=1).mean()]

        if theta_log:
            for i in range(self.d):
                log_list += ['%f' % self.theta[i,j] for j in range(self.C[i])]
        return log_list

    def load_theta_from_log(self, theta_log):
        self.theta = np.zeros((self.d, self.Cmax))
        k = 0
        for i in range(self.d):
            for j in range(self.C[i]):
                self.theta[i,j] = theta_log[k]
                k += 1


