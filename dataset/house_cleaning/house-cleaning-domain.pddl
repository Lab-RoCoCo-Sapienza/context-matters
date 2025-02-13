(define (domain house-cleaning-domain)

    (:requirements
        :strips
        :typing
    )

    (:types
        room locatable - object
        robot grabbable bin - locatable
        disposable mop - grabbable
    )

    (:predicates
        (at ?something - locatable ?where - room)
        (is-holding ?who - robot ?something - grabbable)
        (is-free ?who - robot)
        (thrashed ?what - disposable)
        (is-clean ?what - mop)
        (is-dirty ?what - mop)
        (dirty-floor ?what - room)
        (clean-floor ?what - room)
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
    
    (:action throw_away
        :parameters (?who - robot ?what - disposable ?in - bin ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?what) (at ?in ?where))
        :effect (and (not (is-holding ?who ?what)) (is-free ?who) (thrashed ?what) (not (at ?what ?where)))
    )
    
    (:action mop_floor
        :parameters (?who - robot ?with - mop ?where - room)
        :precondition (and (at ?who ?where) (is-holding ?who ?with)
                    (is-clean ?with) (dirty-floor ?where))
        :effect (and (not (dirty-floor ?where)) (not (is-clean ?with))
                (clean-floor ?where) (is-dirty ?with)    
        )
    )
    
    (:action clean_mop
        :parameters (?who - robot ?what - mop)
        :precondition (and (is-holding ?who ?what) (is-dirty ?what))
        :effect (and (not (is-dirty ?what)) (is-clean ?what))
    )
)
