(define (domain laundry-domain)

    (:requirements
        :strips
        :typing
    )

    (:types
        room locatable - object
        robot washing-machine grabbable - locatable
        cleaning-supply cleanable - grabbable
    )

    (:predicates
        (at ?something - locatable ?where - room)
        (is-holding ?who - robot ?something - grabbable)
        (is-free ?who - robot)
        (is-dirty ?what - cleanable)
        (is-clean ?what - cleanable)
        (is-open ?what - washing-machine)
        (is-closed ?what - washing-machine)
        (is-refilled ?what - washing-machine)
        (is-empty ?what - washing-machine)
        (inside ?what - cleanable ?where - washing-machine)
    )

    (:action move_to
        :parameters (?who - robot ?from - room ?to - room)
        :precondition (and (at ?who ?from))
        :effect (and (not (at ?who ?from)) (at ?who ?to))
    )
    
    (:action grab
        :parameters (?who - robot ?what - grabbable ?where - room)
        :precondition (and (at ?who ?where) (at ?what ?where) (is-free ?who))
        :effect (and (not (at ?what ?where)) (is-holding ?who ?what) (not (is-free ?who)))
    )
    
    (:action drop
        :parameters (?who - robot ?what - grabbable ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?what))
        :effect (and (not (is-holding ?who ?what)) (is-free ?who) (at ?what ?where))
    )
    
    (:action open
        :parameters (?who - robot ?what - washing-machine ?where - room)
        :precondition (and (at ?who ?where) (at ?what ?where) (is-closed ?what))
        :effect (and (is-open ?what) (not (is-closed ?what)))
    )
    
    (:action close
        :parameters (?who - robot ?what - washing-machine ?where - room)
        :precondition (and (at ?who ?where) (at ?what ?where) (is-open ?what))
        :effect (and (is-closed ?what) (not (is-open ?what)))
    )
    
    (:action refill
        :parameters (?who - robot ?what - washing-machine ?where - room ?with - cleaning-supply)
        :precondition (and (at ?who ?where) (at ?what ?where) (is-holding ?who ?with) (is-empty ?what))
        :effect (and (is-refilled ?what) (not (is-empty ?what)))
    )
    
    (:action put_inside
        :parameters (?who - robot ?what - cleanable ?in - washing-machine ?where - room) 
        :precondition (and (at ?who ?where) (at ?in ?where) (is-holding ?who ?what) (is-open ?in))
        :effect (and (not (is-holding ?who ?what)) (inside ?what ?in) (is-free ?who))
    )
    
    (:action wash
        :parameters (?who - robot ?what - cleanable ?in - washing-machine ?where - room) 
        :precondition (and (at ?who ?where) (at ?in ?where) (inside ?what ?in) (is-closed ?in) (is-refilled ?in))
        :effect (and (is-clean ?what) (not (is-dirty ?what)))
    )
    
)

