# DROP TABLE `nlp_list_person_helper`;

CREATE TABLE `nlp_list_person_helper` (
	`ID` INT(11) NOT NULL AUTO_INCREMENT,
	`DesignID` INT(11) NULL DEFAULT NULL,
	`NLPID` INT(11) NULL DEFAULT NULL,
	PRIMARY KEY (`ID`),
	INDEX `Design_FK` (`DesignID`),
	INDEX `NLP_FK` (`NLPID`),
	CONSTRAINT `Design_FK` FOREIGN KEY (`DesignID`) REFERENCES `Designs` (`DesignID`),
	CONSTRAINT `NLP_FK` FOREIGN KEY (`NLPID`) REFERENCES `nlp_list_person` (`id`)
)
COLLATE='utf8_general_ci'
ENGINE=InnoDB
;

# includes the mapping to alternativenames
INSERT INTO nlp_list_person_helper(DesignID, NLPID)
SELECT ner.DesignID, nlp.id FROM cnt_pipeline_ner ner, nlp_list_person nlp WHERE
LABEL_ENTITY = 'PERSON' AND
(ner.entity = nlp.name or INSTR(nlp.alternativenames, ner.entity) > 0); 

# not found "PERSON" can be listed by:
select distinct Entity from cnt_pipeline_ner ner WHERE
LABEL_ENTITY = 'PERSON' and 
Entity not in 
(select distinct Entity from cnt_pipeline_ner, nlp_list_person nlp WHERE
LABEL_ENTITY = 'PERSON' AND
(ner.entity = nlp.name or INSTR(nlp.alternativenames, ner.entity) > 0));
