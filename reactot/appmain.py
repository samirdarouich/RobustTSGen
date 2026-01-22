from yarp.parsers import xyz_parse
from yarp.utils import opt_geo
from yarp.taffi_functions import table_generator
from yarp.find_lewis import find_lewis
from yarp.parsers import xyz_write
from run_model import pred_ts
import argparse


def modify_radj_mat(Radj_mat, bond_break, bond_form):
    # Convert bond_break and bond_form from string to list of tuples
    bond_break = eval(bond_break)
    bond_form = eval(bond_form)
    
    for (i, j) in bond_break:
        Radj_mat[i-1][j-1] = 0  
        Radj_mat[j-1][i-1] = 0


    for (i, j) in bond_form:
        Radj_mat[i-1][j-1] = 1
        Radj_mat[j-1][i-1] = 1

    return Radj_mat

def main(opt):
    if opt.bond_break and opt.bond_form:
        [[E,G]] = xyz_parse(opt.rxyz, multiple=True)
        R_adj_mat = table_generator(E,G)
        P_adj_mat = modify_radj_mat(R_adj_mat, opt.bond_break, opt.bond_form)
        P_bond_mats, scores = find_lewis(E, P_adj_mat)
        P_bond_mat = P_bond_mats[0]
        P_G1 = opt_geo(E, G, P_bond_mat)
        xyz_write("product.xyz",E,P_G1)
        pred_ts(opt.rxyz, "product.xyz", opt, opt.output_path)
    elif opt.pxyz:
        pred_ts(opt.rxyz, opt.pxyz, opt, opt.output_path)
    else:
        raise ValueError("Either --bond_break and --bond_form or --pxyz must be provided")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rxyz', type=str, help='Specify the input file path')
    parser.add_argument('--bond_break', type=str, help='Specify the bonds to break as a list of tuples')
    parser.add_argument('--bond_form', type=str, help='Specify the bonds to form as a list of tuples')
    parser.add_argument("--pxyz", type=str, default='', help='Specify the product file path')
    parser.add_argument("--output_path",    type=str)
    
    parser.add_argument("--batch-size",     type=int,   default=72)
    parser.add_argument("--nfe",            type=int,   default=100)

    parser.add_argument("--solver",           type=str, default='ddpm', choices=["ddpm", "ei", "ode"])
    parser.add_argument("--checkpoint_path",  type=str, default='/root/react-ot/reactot-pretrained.ckpt')

    # ei
    parser.add_argument("--order",          type=int, default=1)
    parser.add_argument("--diz",            type=str, default="linear", choices=["linear", "quad"])

    # ode
    parser.add_argument("--method",         type=str,   default="midpoint")
    parser.add_argument("--atol",           type=float, default=1e-2)
    parser.add_argument("--rtol",           type=float, default=1e-2)
    
    opt = parser.parse_args()

    # Ensure --pxyz cannot be used with --bond_break and --bond_form
    if opt.pxyz and (opt.bond_break or opt.bond_form):
        parser.error("--pxyz cannot be used with --bond_break or --bond_form")

    main(opt)

