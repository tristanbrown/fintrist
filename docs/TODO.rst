#####
TO DO
#####

Add
---
* Implement a neural network-training Study. 

Fix
---

Change
------
* Make parent studies spawn automatically. Process will need parent info.
 - Trying to schedule a study should:
    1. Identify the names of parents to look for.
    2. Query or create the relevant parent, with the appropriate process and symbol.
        - Parents must be named systematically.
        - Parameters like `symbol` might need to be passed up the inheritance chain.
        - Parent info will need to be stored on the Process. This could come from:
            A. The function's docstring?
                Pros: Easy to edit and see.
                Cons: Bad practice? Less reliable?
            B. The function could have a "return metadata" short-circuit?
                Pros: Simple code.
                Cons: Boilerplate. Blending functional and object-oriented programming.
            C. Just keep it on the Process db object?
                Pros: Storing metadata on database objects is pretty standard. Customizable.
                Cons: Not as visible. More easily corrupted.
            D. Combine (C) with (A)?
        - Can Process objects and Recipe objects be combined such that registering a process
            creates a new Recipe?

    3. Check the validity of the parent's data. 
    4. Run the study. 
