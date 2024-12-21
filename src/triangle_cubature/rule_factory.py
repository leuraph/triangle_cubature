import numpy as np
from triangle_cubature.cubature_rule \
    import CubatureRule, CubatureRuleEnum, WeightsAndIntegrationPoints


def get_rule(rule: CubatureRuleEnum) -> CubatureRule:
    """
    given a cubature rule, returns the corresponding
    weight(s) and integration point(s)

    Notes
    -----
    - the rules correspond to the rules as specified in [1]

    References
    ----------
    - [1] Stenger, Frank.
      'Approximate Calculation of Multiple Integrals (A. H. Stroud)'.
      SIAM Review 15, no. 1 (January 1973): 234-35.
      https://doi.org/10.1137/1015023. p. 306-315
    """
    if rule == CubatureRuleEnum.MIDPOINT:
        weights = np.array([1./2.])
        integration_points = np.array([1./3., 1./3.]).reshape(1, 2)
        weights_and_integration_points = WeightsAndIntegrationPoints(
            weights=weights,
            integration_points=integration_points)
        name = 'midpoint'
        degree_of_exactness = 1

        return CubatureRule(
            weights_and_integration_points=weights_and_integration_points,
            degree_of_exactness=degree_of_exactness,
            name=name)

    if rule == CubatureRuleEnum.LAUFFER_LINEAR:
        integration_points = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.]
        ])
        weights = np.array([
            1/3 * 0.5,
            1/3 * 0.5,
            1/3 * 0.5
        ])
        weights_and_integration_points = WeightsAndIntegrationPoints(
            weights=weights,
            integration_points=integration_points)
        name = 'lauffer-linear'
        degree_of_exactness = 1

        return CubatureRule(
            weights_and_integration_points=weights_and_integration_points,
            degree_of_exactness=degree_of_exactness,
            name=name)

    if rule == CubatureRuleEnum.SMPLX1:
        r = (1.)/(6.)
        s = 2./3.
        integration_points = np.array([
            [r, r],
            [r, s],
            [s, r]
        ])
        weights = np.array([
            1/3 * 0.5,
            1/3 * 0.5,
            1/3 * 0.5
        ])
        weights_and_integration_points = WeightsAndIntegrationPoints(
            weights=weights,
            integration_points=integration_points)
        name = 'SMPLX1'
        degree_of_exactness = 2

        return CubatureRule(
            weights_and_integration_points=weights_and_integration_points,
            degree_of_exactness=degree_of_exactness,
            name=name)

    if rule == CubatureRuleEnum.DAYTAYLOR:
        """
        https://www.math.unipd.it/~alvise/SETS_CUBATURE_TRIANGLE/day_taylor/set_day_taylor_standard.m

        If specified lik unavailable, see instead
        D.M. Day and M.A. Taylor,
        "A new 11 point degree 6 formula for the triangle",
        PAMM Proc. Appl. Math. Mech. 7 1022501-1022502 (2007).
        """

        integration_points = np.array([
            [5.72549866774768601018763547472190e-02,
                8.95498146789879490015096052957233e-01],
            [8.95362640024579103936730462010019e-01,
                6.18282212503219533172860167269391e-02],
            [6.84475748456514043738252439652570e-01,
                2.33437384976827311255931363120908e-02],
            [6.87462559150295304810640573123237e-02,
                6.00302757472630024726534259116306e-02],
            [6.15676205575839574635210738051683e-01,
                3.33461808341377174969011321081780e-01],
            [6.27946141197789464705181217141217e-01,
                1.59189185992151482906820092466660e-01],
            [6.29091383418635685664810353046050e-02,
                6.55295093705452469379224567092024e-01],
            [6.83782119205099125913704938284354e-02,
                3.09117685428267230385301900241757e-01],
            [2.87529458374392254960127957019722e-01,
                6.36426509179620181200220940809231e-01],
            [3.28783556413134614437865366198821e-01,
                7.70240056424634222942415817669826e-02],
            [3.12290405013644800646943622268736e-01,
                3.52344786445899504911949406960048e-01]])
        weights = np.array([
            1.90340359264777984893424189749567e-02,
            1.91896776538764100850098515138598e-02,
            2.31002283722809183263979804223709e-02,
            2.67337947220994999464327435134692e-02,
            4.18779134828728416550802648998797e-02,
            5.08224165127585322809800061349961e-02,
            5.09307622306834767433869615160802e-02,
            5.57109158300008525110946777658683e-02,
            5.60047251314730321070101126679219e-02,
            6.23937857187791614088645530955546e-02,
            9.42017444186974139963552943299874e-02])
        weights_and_integration_points = WeightsAndIntegrationPoints(
            weights=weights,
            integration_points=integration_points)
        name = 'DAYTAYLOR'
        degree_of_exactness = 6

        return CubatureRule(
            weights_and_integration_points=weights_and_integration_points,
            degree_of_exactness=degree_of_exactness,
            name=name)

    raise ValueError('specified rule does not exist.')
