(define (domain allensville)
    (:requirements :strips :typing :negative-preconditions)
    (:types location object fixture appliance vacuum mop)

    (:predicates 
        (is_grabbable ?obj - object) 
        (is_located ?obj - object ?loc - location) 
        (is_clean ?obj) 
        (is_cleanable ?obj)
        (holding ?obj)
        (arm_is_free)
        (robot_at ?loc - location)
    )

    (:action pick
        :parameters (?obj - object ?loc - location)
        :precondition (and 
            (is_grabbable ?obj)
            (arm_is_free)
            (robot_at ?loc)
            (is_located ?obj ?loc)
        )
        :effect (and 
            (not (arm_is_free))
            (not (is_located ?obj ?loc))
            (holding ?obj)

        )
    )

    (:action place
        :parameters (?obj - object ?loc - location)
        :precondition (and 
            (not (arm_is_free))
            (holding ?obj)
            (robot_at ?loc)
        )
        :effect (and 
            (arm_is_free)
            (not (holding ?obj))
            (is_located ?obj ?loc)
        )
    )

    (:action move
        :parameters (?from - location ?to - location)
        :precondition (and 
            (robot_at ?from)
        )
        :effect (and 
            (not (robot_at ?from))
            (robot_at ?to)
        )
    )

    (:action clean_object
        :parameters (?dev - device ?obj - object ?loc - location)
        :precondition (and 
            (robot_at ?loc)
            (is_located ?obj ?loc)
            (holding ?obj)
            (not (is_clean ?obj))
            (is_cleanable ?obj)
        )
        :effect (and 
            (is_clean ?obj)
        )
    )
    
)