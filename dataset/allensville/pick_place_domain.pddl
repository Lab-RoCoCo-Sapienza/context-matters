(define (domain pick_place_domain)
    (:requirements :strips :typing :negative-preconditions)
    (:types location object fixture appliance)

    (:predicates 
        (is_grabbable ?obj - object) 
        (is_located ?obj - object ?loc - location) 
        (is_clean ?obj) 
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
)