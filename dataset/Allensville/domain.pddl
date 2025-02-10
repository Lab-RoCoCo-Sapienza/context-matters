(define (domain allensville)
    (:requirements :strips :typing)
    
    (:types
        location
        vehicle
        package
    )
    
    (:predicates
        (at ?loc - location)
        (in ?pkg - package ?veh - vehicle)
        (delivered ?pkg - package)
    )
    
    (:action move
        :parameters (?from - location ?to - location ?veh - vehicle)
        :precondition (at ?from)
        :effect (and (not (at ?from)) (at ?to))
    )
    
    (:action load
        :parameters (?pkg - package ?loc - location ?veh - vehicle)
        :precondition (and (at ?loc) (not (in ?pkg ?veh)))
        :effect (in ?pkg ?veh)
    )
    
    (:action unload
        :parameters (?pkg - package ?loc - location ?veh - vehicle)
        :precondition (and (at ?loc) (in ?pkg ?veh))
        :effect (and (not (in ?pkg ?veh)) (delivered ?pkg))
    )
)