# Tpetra Wrappers for deal.II    
  
This GitHub repository contains the basic Tpetra wrappers, which act as an interface from deal.II to Trilinos.    
The contributions of the TpetraWrappers are listed below.   
Note that the namespace prefix LinearAlgebra:: was neglected on all mentioned TpetraWrappers symbols for readability:   
- An extension of the TpetraWrappers::Vector   
- The TpetraWrappers::Vector class has been overhauled   
- A first version of TpetraWrappers::SparseMatrix and TpetraWrappers::SparsityPattern have been implemented, mirroring the functionality of the corresponding Epetra-based classes.    
- The framework for block classes was added, including TpetraWrappers::BlockVector and the TpetraWrappers::BlockSparseMatrix.    
  
The starting point for this work was the branch [dealii-9.5](https://github.com/dealii/dealii/tree/dealii-9.5) (based on the commit: 9e847302b21355f355c87890477f4dd485da26b1).   
  
The Tpetra Wrappers were further developed by the other deal.II developers and are part of the deal.II version 9.6.    
A patch to the cmake files was added so this preview would work.
